#!/bin/bash
# Fix DNABERT-2's flash_attn_triton.py for Triton 3.x compatibility.
#
# Triton 3.x removed the `trans_b` kwarg from `tl.dot()`.
# This script replaces every occurrence with `tl.trans()`.

set -e

echo "=========================================="
echo "DNABERT-2 Triton Compatibility Fix"
echo "=========================================="
echo

CACHE_DIR="$HOME/.cache/huggingface/modules/transformers_modules"

# Find ALL copies (HF cache may have both "DNABERT-2-117M" and "DNABERT_hyphen_2_hyphen_117M")
mapfile -t FILES < <(find "$CACHE_DIR" -name "flash_attn_triton.py" 2>/dev/null | grep -i dnabert)

if [ ${#FILES[@]} -eq 0 ]; then
    echo "Error: Could not find any flash_attn_triton.py for DNABERT-2"
    echo "Please ensure you have loaded the DNABERT-2 model at least once."
    exit 1
fi

echo "Found ${#FILES[@]} copy/copies:"
printf "  %s\n" "${FILES[@]}"
echo

TOTAL_FIXED=0
for FILE in "${FILES[@]}"; do
    COUNT=$(grep -c 'trans_b=True' "$FILE" 2>/dev/null || true)
    if [ "$COUNT" -eq 0 ]; then
        echo "  $FILE — already patched"
        continue
    fi
    echo "  $FILE — fixing $COUNT occurrence(s)..."
    sed -i 's/tl\.dot(\([^,]*\), \([^,]*\), trans_b=True)/tl.dot(\1, tl.trans(\2))/g' "$FILE"
    REMAINING=$(grep -c 'trans_b=True' "$FILE" 2>/dev/null || true)
    if [ "$REMAINING" -ne 0 ]; then
        echo "    Warning: $REMAINING instance(s) remain:"
        grep -n 'trans_b=True' "$FILE"
    else
        echo "    All $COUNT fixed."
    fi
    TOTAL_FIXED=$((TOTAL_FIXED + COUNT))
done

echo
echo "Clearing Triton cache..."
rm -rf "$HOME/.triton/cache"/* 2>/dev/null || true
mkdir -p "$HOME/.triton/cache" 2>/dev/null || true
echo "Triton cache cleared."

echo
echo "=========================================="
echo "Patch complete!"
echo "=========================================="
