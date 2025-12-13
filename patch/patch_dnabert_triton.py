#!/usr/bin/env python3
"""
Patch script to fix DNABERT-2 Triton compatibility issue.

This script fixes the 'trans_b' parameter issue in flash_attn_triton.py
by replacing tl.dot(..., trans_b=True) with tl.dot(..., tl.trans(...)).

Usage:
    python patch_dnabert_triton.py
"""

import os
import re
from pathlib import Path


def find_flash_attn_triton_file():
    """Find the flash_attn_triton.py file in HuggingFace cache."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
    
    # Look for DNABERT-2 model files
    pattern = "**/zhihan1996/DNABERT*/**/flash_attn_triton.py"
    
    for file_path in cache_dir.glob(pattern):
        if file_path.exists():
            return file_path
    
    # Alternative: search more broadly
    for file_path in cache_dir.glob("**/flash_attn_triton.py"):
        if "DNABERT" in str(file_path) or "dnabert" in str(file_path).lower():
            return file_path
    
    return None


def patch_file(file_path):
    """Apply patches to fix trans_b compatibility issues."""
    if not file_path or not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return False
    
    print(f"Reading file: {file_path}")
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Fix 1: qk += tl.dot(q, k, trans_b=True) -> qk += tl.dot(q, tl.trans(k))
    content = re.sub(
        r'qk\s*\+=\s*tl\.dot\(q,\s*k,\s*trans_b=True\)',
        'qk += tl.dot(q, tl.trans(k))',
        content
    )
    
    # Fix 2: qk = tl.dot(q, k, trans_b=True) -> qk = tl.dot(q, tl.trans(k))
    content = re.sub(
        r'qk\s*=\s*tl\.dot\(q,\s*k,\s*trans_b=True\)',
        'qk = tl.dot(q, tl.trans(k))',
        content
    )
    
    # Fix 3: dp = tl.dot(do, v, trans_b=True) -> dp = tl.dot(do, tl.trans(v))
    content = re.sub(
        r'dp\s*=\s*tl\.dot\(do,\s*v,\s*trans_b=True\)',
        'dp = tl.dot(do, tl.trans(v))',
        content
    )
    
    # Check if any changes were made
    if content == original_content:
        # Check if already patched
        if 'tl.trans(' in content and 'trans_b=True' not in content:
            print("File appears to already be patched.")
            return True
        else:
            print("Warning: No changes made. File might not contain trans_b=True patterns.")
            return False
    
    # Write the patched content
    print(f"Writing patched file: {file_path}")
    with open(file_path, 'w') as f:
        f.write(content)
    
    # Verify the patch
    if 'trans_b=True' in content:
        print("Warning: Some trans_b=True patterns may still remain. Please check manually.")
        return False
    
    print("✓ Successfully patched file!")
    return True


def clear_caches():
    """Clear Triton and Python caches."""
    import shutil
    
    # Clear Triton cache
    triton_cache = Path.home() / ".triton" / "cache"
    if triton_cache.exists():
        print(f"Clearing Triton cache: {triton_cache}")
        shutil.rmtree(triton_cache, ignore_errors=True)
        triton_cache.mkdir(parents=True, exist_ok=True)
        print("✓ Triton cache cleared")
    
    # Clear Python cache for the module
    cache_dir = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
    for pycache in cache_dir.rglob("__pycache__"):
        if "DNABERT" in str(pycache) or "dnabert" in str(pycache).lower():
            print(f"Clearing Python cache: {pycache}")
            shutil.rmtree(pycache, ignore_errors=True)
    
    print("✓ Caches cleared")


def main():
    """Main function to apply the patch."""
    print("=" * 60)
    print("DNABERT-2 Triton Compatibility Patch")
    print("=" * 60)
    print()
    
    # Find the file
    print("Searching for flash_attn_triton.py...")
    file_path = find_flash_attn_triton_file()
    
    if not file_path:
        print("Error: Could not find flash_attn_triton.py in HuggingFace cache.")
        print("Please ensure you have loaded the DNABERT-2 model at least once.")
        print("Expected location: ~/.cache/huggingface/modules/transformers_modules/")
        return 1
    
    print(f"Found file: {file_path}")
    print()
    
    # Apply patch
    if not patch_file(file_path):
        return 1
    
    print()
    
    # Clear caches
    print("Clearing caches...")
    clear_caches()
    
    print()
    print("=" * 60)
    print("Patch applied successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Restart your Python kernel/Jupyter notebook")
    print("2. Reload the DNABERT-2 model")
    print("3. The model should now work with the updated Triton API")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())

