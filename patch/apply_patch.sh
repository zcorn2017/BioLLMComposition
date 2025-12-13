#!/bin/bash
# Script to apply the DNABERT-2 Triton compatibility patch

set -e

echo "=========================================="
echo "DNABERT-2 Triton Compatibility Patch"
echo "=========================================="
echo

# Find the flash_attn_triton.py file
CACHE_DIR="$HOME/.cache/huggingface/modules/transformers_modules"
FILE=$(find "$CACHE_DIR" -path "*/zhihan1996/DNABERT*/**/flash_attn_triton.py" 2>/dev/null | head -1)

if [ -z "$FILE" ]; then
    # Try broader search
    FILE=$(find "$CACHE_DIR" -name "flash_attn_triton.py" | grep -i dnabert | head -1)
fi

if [ -z "$FILE" ]; then
    echo "Error: Could not find flash_attn_triton.py"
    echo "Please ensure you have loaded the DNABERT-2 model at least once."
    exit 1
fi

echo "Found file: $FILE"
echo

# Get the directory containing the file
FILE_DIR=$(dirname "$FILE")
PATCH_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/dnabert_triton_fix.patch"

if [ ! -f "$PATCH_FILE" ]; then
    echo "Error: Patch file not found: $PATCH_FILE"
    exit 1
fi

echo "Applying patch..."
cd "$FILE_DIR"

# Apply the patch
if patch -p1 < "$PATCH_FILE"; then
    echo "✓ Patch applied successfully!"
else
    echo "Error: Failed to apply patch"
    echo "The file may have already been patched or has been modified."
    exit 1
fi

echo

# Clear Triton cache
echo "Clearing Triton cache..."
rm -rf "$HOME/.triton/cache"/* 2>/dev/null || true
mkdir -p "$HOME/.triton/cache" 2>/dev/null || true
echo "✓ Triton cache cleared"

echo
echo "=========================================="
echo "Patch applied successfully!"
echo "=========================================="
echo
echo "Next steps:"
echo "1. Restart your Python kernel/Jupyter notebook"
echo "2. Reload the DNABERT-2 model"
echo "3. The model should now work with the updated Triton API"
echo

