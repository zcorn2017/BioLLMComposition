# DNABERT-2 Triton Compatibility Patch

## Problem

DNABERT-2 uses a custom flash attention implementation that relies on an older Triton API. Newer versions of Triton removed the `trans_b` parameter from `tl.dot()`, causing the following error:

```
TypeError: dot() got an unexpected keyword argument 'trans_b'
```

## Solution

This patch script automatically fixes the compatibility issue by:
1. Finding the `flash_attn_triton.py` file in the HuggingFace cache
2. Replacing all instances of `tl.dot(..., trans_b=True)` with `tl.dot(..., tl.trans(...))`
3. Clearing Triton and Python caches to force recompilation

## Usage

### Option 1: Apply the .patch file (Recommended)

Using the standard `patch` command:

```bash
# Find the file
FILE=$(find ~/.cache/huggingface/modules/transformers_modules -path "*/zhihan1996/DNABERT*/**/flash_attn_triton.py" | head -1)

# Apply the patch
cd $(dirname "$FILE")
patch -p1 < /path/to/dnabert_triton_fix.patch
```

Or use the provided shell script:

```bash
·
```

### Option 2: Run the Python patch script

```bash
python patch_dnabert_triton.py
```

Or make it executable and run directly:

```bash
chmod +x patch_dnabert_triton.py
./patch_dnabert_triton.py
```

### Option 3: Manual patch (if scripts don't work)

1. Find the file:
   ```bash
   find ~/.cache/huggingface/modules/transformers_modules -name "flash_attn_triton.py" | grep -i dnabert
   ```

2. Edit the file and replace:
   - `qk += tl.dot(q, k, trans_b=True)` → `qk += tl.dot(q, tl.trans(k))`
   - `qk = tl.dot(q, k, trans_b=True)` → `qk = tl.dot(q, tl.trans(k))`
   - `dp = tl.dot(do, v, trans_b=True)` → `dp = tl.dot(do, tl.trans(v))`

3. Clear caches:
   ```bash
   rm -rf ~/.triton/cache/*
   ```

## After Patching

**Important:** You must restart your Python kernel/Jupyter notebook after applying the patch for the changes to take effect.

1. Restart your Python kernel
2. Reload the DNABERT-2 model:
   ```python
   from transformers import AutoTokenizer, AutoModel
   tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
   model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
   ```

## What the Patch Does

The patch fixes three instances in `flash_attn_triton.py`:

1. **Line ~191**: `qk += tl.dot(q, k, trans_b=True)` → `qk += tl.dot(q, tl.trans(k))`
2. **Line ~434**: `qk = tl.dot(q, k, trans_b=True)` → `qk = tl.dot(q, tl.trans(k))`
3. **Line ~501**: `dp = tl.dot(do, v, trans_b=True)` → `dp = tl.dot(do, tl.trans(v))`

## Notes

- The patch modifies files in the HuggingFace cache directory
- If you re-download the model or clear the cache, you may need to reapply the patch
- This patch is specific to DNABERT-2 models that use the custom flash attention implementation

## Verification

After patching, verify the fix by checking the file:

```bash
grep -n "trans_b" ~/.cache/huggingface/modules/transformers_modules/zhihan1996/DNABERT*/**/flash_attn_triton.py
```

This should return no results if the patch was successful.

## Files Included

- **`dnabert_triton_fix.patch`** - Standard unified diff patch file (can be applied with `patch` command)
- **`apply_patch.sh`** - Shell script to automatically find and apply the patch
- **`patch_dnabert_triton.py`** - Python script alternative for applying the patch
- **`PATCH_README.md`** - This documentation file

The `.patch` file is the most portable and standard format, compatible with version control systems and can be easily shared or included in repositories.

