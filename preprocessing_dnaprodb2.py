import json
import math
from collections import Counter, defaultdict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# --------------------------------------------------------------------------------------
# 1. Normalize Mongo extended JSON
# --------------------------------------------------------------------------------------

def normalize_mongo_extended(value):
    """
    Recursively normalize MongoDB extended JSON into plain Python
    types so pyarrow can infer a clean schema.

    Handles things like:
      {"$oid": "..."}          -> "..."  (str)
      {"$numberInt": "1"}      -> 1      (int)
      {"$numberDouble": "1.5"} -> 1.5    (float)
    and recurses into lists/dicts.
    """
    if isinstance(value, dict):
        if "$oid" in value:
            return value["$oid"]
        if "$numberInt" in value:
            return int(value["$numberInt"])
        if "$numberDouble" in value:
            s = value["$numberDouble"]
            if s in ("NaN", "nan"):
                return math.nan
            if s in ("Infinity", "inf"):
                return math.inf
            if s in ("-Infinity", "-inf"):
                return -math.inf
            return float(s)
        if "$date" in value:
            # You can later parse to datetime if you want
            return value["$date"]

        # Generic dict: recurse on values
        return {k: normalize_mongo_extended(v) for k, v in value.items()}

    elif isinstance(value, list):
        return [normalize_mongo_extended(v) for v in value]

    else:
        # primitive: int/float/str/None/etc
        return value


data_path = "/home/zcorn/Projects/proteinDNA_data/working/dnaprodb2/dna-protein.json"
out_path = "/home/zcorn/Projects/proteinDNA_data/working/dnaprodb2/dna_protein.parquet"

records = []
with open(data_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        norm = normalize_mongo_extended(raw)
        records.append(norm)

print("JSON loaded and normalized!")

# Optional: DataFrame for quick inspection
df = pd.DataFrame(records)
print("DataFrame created!")
print(df.head())


# --------------------------------------------------------------------------------------
# 2. Strict nested schema inference
# --------------------------------------------------------------------------------------

# Only strictly clean these nested top-level columns
NESTED_COLS = ["dna", "interfaces", "meta_data", "search"]

TYPE_DICT = "dict"
TYPE_LIST = "list"
TYPE_INT = "int"
TYPE_FLOAT = "float"
TYPE_STR = "str"
TYPE_BOOL = "bool"
TYPE_NULL = "null"
TYPE_OTHER = "other"


def classify_type(v):
    if v is None:
        return TYPE_NULL
    if isinstance(v, bool):
        return TYPE_BOOL
    if isinstance(v, dict):
        return TYPE_DICT
    if isinstance(v, list):
        return TYPE_LIST
    if isinstance(v, int):
        return TYPE_INT
    if isinstance(v, float):
        return TYPE_FLOAT
    if isinstance(v, str):
        return TYPE_STR
    return TYPE_OTHER


def gather_type_stats(value, path, stats):
    t = classify_type(value)
    stats[path][t] += 1

    if t == TYPE_DICT:
        for k, v in value.items():
            gather_type_stats(v, path + (k,), stats)
    elif t == TYPE_LIST:
        for elem in value:
            gather_type_stats(elem, path + ("<elem>",), stats)


def choose_canonical_type(counter: Counter):
    # Ignore null-only paths
    non_null = {t: c for t, c in counter.items() if t != TYPE_NULL}
    if not non_null:
        return None

    # Prefer structured types if they are the only non-null type
    if non_null.get(TYPE_DICT, 0) > 0 and len(non_null) == 1:
        return TYPE_DICT
    if non_null.get(TYPE_LIST, 0) > 0 and len(non_null) == 1:
        return TYPE_LIST

    # Numeric preference
    if non_null.get(TYPE_FLOAT, 0) or (
        non_null.get(TYPE_INT, 0) and len(non_null) == 1
    ):
        # if any float or mix of int/float -> float
        return TYPE_FLOAT if non_null.get(TYPE_FLOAT, 0) else TYPE_INT

    if non_null.get(TYPE_INT, 0) and len(non_null) == 1:
        return TYPE_INT

    # Fallbacks: string, bool, other
    if non_null.get(TYPE_STR, 0):
        return TYPE_STR
    if non_null.get(TYPE_BOOL, 0):
        return TYPE_BOOL

    # If truly mixed weird stuff, pick the majority
    return max(non_null.items(), key=lambda kv: kv[1])[0]


def infer_canonical_types_for_column(records, col):
    stats = defaultdict(Counter)
    for r in records:
        v = r.get(col)
        if v is not None:
            gather_type_stats(v, (col,), stats)

    canonical = {}
    for path, counter in stats.items():
        ct = choose_canonical_type(counter)
        if ct is not None:
            canonical[path] = ct
    return canonical


def find_empty_struct_paths(canonical_types):
    """
    Find dict-typed paths that have no child paths.
    These would be empty structs that Parquet cannot write.
    """
    dict_paths = [p for p, t in canonical_types.items() if t == TYPE_DICT]
    has_child = {p: False for p in dict_paths}

    for p in dict_paths:
        for q in canonical_types.keys():
            if len(q) > len(p) and q[: len(p)] == p:
                has_child[p] = True
                break

    empty_struct_paths = {p for p, child in has_child.items() if not child}
    return empty_struct_paths


# --------------------------------------------------------------------------------------
# 3. Cleaning based on inferred schema
# --------------------------------------------------------------------------------------

def try_parse_int(s):
    s = s.strip()
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return int(s)
    return None


def try_parse_float(s):
    s = s.strip()
    if s in {"NaN", "nan"}:
        return math.nan
    if s in {"Infinity", "inf"}:
        return math.inf
    if s in {"-Infinity", "-inf"}:
        return -math.inf
    try:
        return float(s)
    except ValueError:
        return None


def clean_nested(value, path, canonical_types, empty_struct_paths, errors, row_idx):
    """
    Recursively coerce value to match canonical_types[path],
    treat "empty struct" paths as None, and collect all mismatches.
    """
    # Special case: empty-struct path -> always None
    if path in empty_struct_paths:
        if value not in (None, {}):
            t = classify_type(value)
            errors[path].append(
                {
                    "row": row_idx,
                    "expected": "empty-dict->None",
                    "actual": t,
                    "value": value,
                }
            )
        return None

    ct = canonical_types.get(path)

    # If we don't have a canonical type for this path, still recurse so children are cleaned.
    if ct is None:
        t = classify_type(value)
        if t == TYPE_DICT:
            return {
                k: clean_nested(
                    v,
                    path + (k,),
                    canonical_types,
                    empty_struct_paths,
                    errors,
                    row_idx,
                )
                for k, v in value.items()
            }
        if t == TYPE_LIST:
            return [
                clean_nested(
                    elem,
                    path + ("<elem>",),
                    canonical_types,
                    empty_struct_paths,
                    errors,
                    row_idx,
                )
                for elem in value
            ]
        return value

    # Null is always allowed
    if value is None:
        return None

    t = classify_type(value)

    # ---- Dicts ----
    if ct == TYPE_DICT:
        if t != TYPE_DICT:
            errors[path].append(
                {"row": row_idx, "expected": ct, "actual": t, "value": value}
            )
            return None
        # Clean children
        return {
            k: clean_nested(
                v,
                path + (k,),
                canonical_types,
                empty_struct_paths,
                errors,
                row_idx,
            )
            for k, v in value.items()
        }

    # ---- Lists ----
    if ct == TYPE_LIST:
        if t != TYPE_LIST:
            errors[path].append(
                {"row": row_idx, "expected": ct, "actual": t, "value": value}
            )
            return None
        return [
            clean_nested(
                elem,
                path + ("<elem>",),
                canonical_types,
                empty_struct_paths,
                errors,
                row_idx,
            )
            for elem in value
        ]

    # ---- Integers ----
    if ct == TYPE_INT:
        if t == TYPE_INT:
            return value
        if t == TYPE_FLOAT and value.is_integer():
            return int(value)
        if t == TYPE_STR:
            parsed = try_parse_int(value)
            if parsed is not None:
                return parsed
            # treat obvious "missing" markers as null
            if value in {"?", "NA", "N/A", ""}:
                return None
        errors[path].append(
            {"row": row_idx, "expected": ct, "actual": t, "value": value}
        )
        return None

    # ---- Floats ----
    if ct == TYPE_FLOAT:
        if t in {TYPE_INT, TYPE_FLOAT} and not isinstance(value, bool):
            return float(value)
        if t == TYPE_STR:
            parsed = try_parse_float(value)
            if parsed is not None:
                return parsed
            if value in {"?", "NA", "N/A", ""}:
                return None
        errors[path].append(
            {"row": row_idx, "expected": ct, "actual": t, "value": value}
        )
        return None

    # ---- Strings ----
    if ct == TYPE_STR:
        if t == TYPE_STR:
            return value
        # Convert anything else to string
        return str(value)

    # ---- Bool ----
    if ct == TYPE_BOOL:
        if t == TYPE_BOOL:
            return value
        if t == TYPE_STR and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        errors[path].append(
            {"row": row_idx, "expected": ct, "actual": t, "value": value}
        )
        return None

    # ---- Fallback for OTHER ----
    # Just return as-is; Arrow may still cope.
    return value


def clean_column(records, col):
    print(f"\n=== Cleaning column: {col} ===")
    canonical_types = infer_canonical_types_for_column(records, col)
    print("Inferred canonical types (sample):")
    for p, t in list(canonical_types.items())[:20]:
        print("  ", " / ".join(p), "->", t)

    empty_struct_paths = find_empty_struct_paths(canonical_types)
    if empty_struct_paths:
        print("\nEmpty-struct paths (will be set to None):")
        for p in empty_struct_paths:
            print("  ", " / ".join(p))

    errors = defaultdict(list)
    cleaned_values = []

    for i, r in enumerate(records):
        v = r.get(col)
        if v is None:
            cleaned_values.append(None)
        else:
            cleaned_values.append(
                clean_nested(
                    v,
                    (col,),
                    canonical_types,
                    empty_struct_paths,
                    errors,
                    row_idx=i,
                )
            )

    # Replace in original records
    for i, r in enumerate(records):
        r[col] = cleaned_values[i]

    # Report errors
    if errors:
        print(f"\n[WARN] Invalid values found in column '{col}':")
        for path, err_list in errors.items():
            print(f"  Path: {' / '.join(path)}")
            print(f"    Total invalid: {len(err_list)}")
            for e in err_list[:10]:  # show first 10 per path
                print(
                    f"    row={e['row']}, expected={e['expected']}, "
                    f"actual={e['actual']}, value={repr(e['value'])}"
                )
            if len(err_list) > 10:
                print(f"    ... ({len(err_list) - 10} more)")
    else:
        print(f"No invalid values in '{col}'")


# --------------------------------------------------------------------------------------
# 4. Apply strict cleaning to nested columns
# --------------------------------------------------------------------------------------

for col in NESTED_COLS:
    if col in df.columns:
        clean_column(records, col)


# --------------------------------------------------------------------------------------
# 5. Global pass: drop any remaining empty dicts at any depth
# --------------------------------------------------------------------------------------

empty_dict_hits = defaultdict(list)  # path -> list[row indices]


def drop_empty_dicts_rec(value, path, row_idx):
    if isinstance(value, dict):
        new_dict = {
            k: drop_empty_dicts_rec(v, path + (k,), row_idx)
            for k, v in value.items()
        }
        # Keep keys even if values are None – we just care about `{}` itself
        if len(new_dict) == 0:
            empty_dict_hits[path].append(row_idx)
            return None
        return new_dict

    elif isinstance(value, list):
        return [
            drop_empty_dicts_rec(elem, path + ("<elem>",), row_idx)
            for elem in value
        ]

    else:
        return value


def drop_all_empty_dicts(records):
    cleaned = []
    for i, r in enumerate(records):
        cleaned.append(drop_empty_dicts_rec(r, (), i))
    return cleaned


print("\nRunning global empty-dict cleanup...")
records = drop_all_empty_dicts(records)

if empty_dict_hits:
    print("\n[INFO] Empty dicts replaced with None at these paths:")
    for path, rows in empty_dict_hits.items():
        print(f"  Path: {' / '.join(path)}")
        print(f"    Count: {len(rows)}")
        print(f"    First rows: {rows[:10]}")
else:
    print("[INFO] No empty dicts found.")


# --------------------------------------------------------------------------------------
# 6. Build Arrow table & write Parquet
# --------------------------------------------------------------------------------------

print("\nCleaning done. Building Arrow table...")
table = pa.Table.from_pylist(records)
pq.write_table(table, out_path, compression="zstd")
print("✓ Parquet saved:", out_path)
