# Why the Attention Mask is Wrong in CompositionModel

## The Problem

In `CompositionModel.forward()`, line 1169:
```python
attn_mask = torch.matmul(
    prot_input["attention_mask"].unsqueeze(2).float(),
    pep_input['attention_mask'].unsqueeze(1).float(),
).repeat(20, 1, 1)
```

This has **two critical issues**:

---

## Issue 1: Wrong Shape for MultiheadAttention

### Current Code:
```python
attn_mask.repeat(20, 1, 1)  # Creates shape: [20, batch_size, seq_len, seq_len]
```

### What PyTorch Expects:

When using `nn.MultiheadAttention(embed_dim=320, num_heads=20, batch_first=True)`, the `attn_mask` parameter expects:

**Option A (Recommended):** `[batch_size, seq_len, seq_len]`
- PyTorch automatically broadcasts this to all 20 heads
- Same mask applied to all heads

**Option B:** `[batch_size * num_heads, seq_len, seq_len]` 
- If you want different masks per head
- Shape would be `[batch_size * 20, seq_len, seq_len]`

### Why `.repeat(20, 1, 1)` is Wrong:

```python
# Input shape after matmul: [batch_size, prot_seq_len, pep_seq_len]
# After .repeat(20, 1, 1): [20, batch_size, prot_seq_len, pep_seq_len]
```

This creates a **4D tensor** with shape `[20, batch, seq, seq]`, but PyTorch expects:
- **3D tensor** with shape `[batch, seq, seq]` OR
- **3D tensor** with shape `[batch*20, seq, seq]`

The current code puts `num_heads` as the **first dimension**, which is incorrect. It should be either:
- Not expanded at all (let PyTorch broadcast), OR
- Expanded to `[batch*20, seq, seq]` if you need per-head masks

---

## Issue 2: Wrong Mask Semantics (Values are Inverted)

### Current Code Creates:
```python
# attention_mask: 1 = valid token, 0 = padding
# After matmul: 1 = both tokens valid, 0 = at least one is padding
attn_mask = torch.matmul(
    prot_input["attention_mask"].unsqueeze(2).float(),  # [batch, prot_len, 1]
    pep_input['attention_mask'].unsqueeze(1).float(),   # [batch, 1, pep_len]
)
# Result: [batch, prot_len, pep_len] where 1.0 = valid, 0.0 = padding
```

### What PyTorch MultiheadAttention Expects:

According to PyTorch documentation, `attn_mask` should be:
- **Boolean or float mask** where:
  - `True` / `1.0` = **MASK OUT** (don't attend, set to -inf)
  - `False` / `0.0` = **ATTEND** (normal attention)

### The Problem:

Your current mask has:
- `1.0` = valid tokens (should attend)
- `0.0` = padding (should mask out)

But PyTorch interprets:
- `1.0` = mask out (don't attend)
- `0.0` = attend

**This is backwards!** You're telling the model to:
- ✅ Attend to padding tokens (`0.0` → attend)
- ❌ Mask out valid tokens (`1.0` → mask out)

---

## The Correct Fix

```python
# Step 1: Create the cross-attention mask
# Shape: [batch_size, prot_seq_len, pep_seq_len]
attn_mask = torch.matmul(
    prot_input["attention_mask"].unsqueeze(2).float(),  # [batch, prot_len, 1]
    pep_input['attention_mask'].unsqueeze(1).float(),   # [batch, 1, pep_len]
)
# Now: 1.0 = both valid, 0.0 = at least one is padding

# Step 2: Invert the mask (1.0 = attend → 0.0, 0.0 = mask → 1.0)
# PyTorch: True/1.0 = mask out, False/0.0 = attend
attn_mask = (1.0 - attn_mask)  # Now: 0.0 = both valid (attend), 1.0 = padding (mask out)

# Step 3: Convert to boolean (optional, but cleaner)
attn_mask = attn_mask.bool()  # True = mask out, False = attend

# Step 4: Shape handling
# Option A: Let PyTorch broadcast (recommended for same mask per head)
# Just pass [batch, seq, seq] - PyTorch handles broadcasting to 20 heads

# Option B: If you need per-head masks (unlikely here)
# attn_mask = attn_mask.unsqueeze(1).repeat(1, 20, 1, 1)  # [batch, 20, seq, seq]
# attn_mask = attn_mask.view(batch_size * 20, seq_len, seq_len)  # [batch*20, seq, seq]
```

### Recommended Solution:

```python
# Create cross-attention mask
attn_mask = torch.matmul(
    prot_input["attention_mask"].unsqueeze(2).float(),
    pep_input['attention_mask'].unsqueeze(1).float(),
)
# Invert: 1.0 = valid → 0.0 (attend), 0.0 = padding → 1.0 (mask out)
attn_mask = (1.0 - attn_mask).bool()  # Shape: [batch, prot_len, pep_len]
# PyTorch will automatically broadcast to 20 heads
```

---

## Why This Matters

1. **Shape Error Risk**: The current `[20, batch, seq, seq]` shape might cause runtime errors or unexpected behavior
2. **Inverted Attention**: The model is currently attending to padding and ignoring valid tokens!
3. **Performance Impact**: Incorrect masking can significantly hurt model performance

---

## Comparison with AttentionModel

Notice that `AttentionModel` (which uses 1 head) does:
```python
attn_mask = torch.matmul(...).repeat(1, 1, 1)  # No-op, keeps [batch, seq, seq]
```

This works because:
- 1 head doesn't need expansion
- But it still has the **inverted mask problem** (not fixed in my review)

The CompositionModel needs special handling because it has **20 heads**, but the current `.repeat(20, 1, 1)` is still wrong.

---

## Summary

**Current code issues:**
1. ❌ Shape: `[20, batch, seq, seq]` → Should be `[batch, seq, seq]` (let PyTorch broadcast)
2. ❌ Values: `1.0` = attend, `0.0` = mask → Should be inverted

**Correct approach:**
1. ✅ Create mask: `[batch, prot_len, pep_len]` with `1.0` = valid, `0.0` = padding
2. ✅ Invert: `(1.0 - mask)` so `0.0` = attend, `1.0` = mask out
3. ✅ Convert to bool: `mask.bool()`
4. ✅ Pass as-is: PyTorch broadcasts to all 20 heads automatically

