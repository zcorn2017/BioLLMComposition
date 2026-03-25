# Code Review: BioLLMComposition.ipynb

## Executive Summary
This notebook implements multiple model architectures for protein-peptide binding prediction. While functional, there are several areas for improvement regarding code quality, reproducibility, error handling, and maintainability.

---

## 🔴 Critical Issues

### 1. **Reproducibility - Missing Random Seeds**
**Location**: Throughout notebook  
**Issue**: No random seeds set for PyTorch, NumPy, or Python's random module  
**Impact**: Results are not reproducible across runs  
**Fix**:
```python
# Add at the beginning of Cell 1
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 2. **Data Loss in Preprocessing**
**Location**: Cell 1, lines 76-95  
**Issue**: Silent failure when splitting sequences - uses bare `except: pass`  
**Impact**: Data is silently dropped without logging  
**Fix**:
```python
# Replace bare except with specific exception handling
mhc_train_pep = []
mhc_train_rec = []
mhc_train_lab = []
dropped_count = 0
for i, s in enumerate(train_seqs):
    try:
        rec, pep = s.split('/')
        mhc_train_pep.append(pep)
        mhc_train_rec.append(rec)
        mhc_train_lab.append(train_labs[i])
    except ValueError:
        dropped_count += 1
        if VERBOSE:
            print(f"Dropped sequence {i}: {s}")
if dropped_count > 0:
    print(f"Warning: Dropped {dropped_count} sequences from training set")
```

### 3. **Incorrect Attention Mask in CompositionModel**
**Location**: Cell 13, line 1158  
**Issue**: `attn_mask.repeat(20, 1, 1)` creates incorrect shape for MultiheadAttention  
**Impact**: Attention mask may not work correctly (should be `[batch, seq_len, seq_len]` or `[batch*num_heads, seq_len, seq_len]`)  
**Fix**:
```python
# MultiheadAttention expects mask shape: [batch*num_heads, seq_len, seq_len] or [batch, seq_len, seq_len]
attn_mask = torch.matmul(
    prot_input["attention_mask"].unsqueeze(2).float(),
    pep_input['attention_mask'].unsqueeze(1).float(),
)
# Invert mask (0 = attend, 1 = mask out) and expand for num_heads
attn_mask = (1 - attn_mask).bool()  # Convert to boolean mask
attn_mask = attn_mask.repeat(20, 1, 1)  # Expand for 20 heads
```

### 4. **Memory Leak - Embeddings on Device**
**Location**: Cell 1, lines 148-151  
**Issue**: Tokenized data moved to device but embeddings are computed on CPU then moved back  
**Impact**: Unnecessary device transfers, potential memory issues  
**Fix**: Keep embeddings on device or move tokens to CPU before tokenization

---

## 🟠 Major Issues

### 5. **Inconsistent Model Saving**
**Location**: Multiple cells  
**Issue**: Some models save checkpoints, composition model has saving commented out  
**Impact**: Cannot reproduce best models  
**Fix**: Uncomment and standardize model saving:
```python
if test_accuracy > best_test_acc:
    best_test_acc = test_accuracy
    torch.save(model.state_dict(), "comp_model.pth")
```

### 6. **Incorrect Visualization Labels**
**Location**: Cells 3, 5 (visualization sections)  
**Issue**: `c=[1]*embs_cat.shape[0]` creates uniform color, labels not used  
**Impact**: Visualizations don't show actual class separation  
**Fix**:
```python
# Use actual labels
scatter = plt.scatter(embs_cat[:, 0], embs_cat[:, 1], 
                      c=mhc_val_lab, cmap=cm, s=10, alpha=0.9)
```

### 7. **Label Mutation in Contrastive Learning**
**Location**: Cell 9, lines 793, 825  
**Issue**: Modifies labels in-place: `labels[labels == 0] = -1`  
**Impact**: Original labels are lost, could cause issues if batch is reused  
**Fix**:
```python
# Create a copy for loss computation
labels_for_loss = labels.clone()
labels_for_loss[labels_for_loss == 0] = -1
loss = criterion(pep_out, pro_out, labels_for_loss)
```

### 8. **Incorrect Attention Model Visualization**
**Location**: Cell 11, line 1059  
**Issue**: Variable name mismatch - uses `pep_out` but should use model output  
**Impact**: Visualization may show wrong data  
**Fix**:
```python
outputs, embeddings = model(pep_tokens, pro_tokens)
for example, y in zip(embeddings, labels):
    embs_attention.append(example.detach().cpu().numpy())
    y_true.append(y.item())
```

### 9. **Hard-coded Magic Numbers**
**Location**: Throughout  
**Issue**: Embedding dimensions (320, 640), hidden sizes (128), max lengths (9, 181) hard-coded  
**Impact**: Difficult to modify or understand  
**Fix**: Define constants at the top:
```python
# Model architecture constants
EMBEDDING_DIM = 320
HIDDEN_DIM = 128
NUM_CLASSES = 2
PEP_MAX_LEN = 9
PROT_MAX_LEN = 181
```

---

## 🟡 Moderate Issues

### 10. **Code Duplication**
**Location**: Training loops in Cells 3, 5, 7, 9, 11, 13  
**Issue**: Nearly identical training/evaluation code repeated  
**Impact**: Hard to maintain, bugs need fixing in multiple places  
**Fix**: Create reusable training function:
```python
def train_model(model, train_loader, val_loader, epochs, device, 
                criterion, optimizer, model_name, verbose=False):
    best_val_acc = 0
    for epoch in tqdm(range(epochs)):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for batch in train_loader:
            # ... training code ...
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                # ... validation code ...
        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), f"{model_name}.pth")
    
    return best_val_acc
```

### 11. **Unused Imports**
**Location**: Cell 1  
**Issue**: `math`, `Adam` (should be `torch.optim.Adam`), `ExponentialLR`, `PCA` imported but not used  
**Impact**: Clutters namespace, confuses readers  
**Fix**: Remove unused imports

### 12. **Inconsistent Variable Naming**
**Location**: Throughout  
**Issue**: Mix of `plm`/`PLM`, `pep`/`P`, inconsistent casing  
**Impact**: Harder to read and maintain  
**Fix**: Use consistent naming convention (e.g., `peptide_model`, `protein_model`)

### 13. **Missing Data Validation**
**Location**: Cell 1  
**Issue**: No checks for empty datasets, shape mismatches, or invalid labels  
**Impact**: Runtime errors could occur later  
**Fix**: Add validation:
```python
assert len(mhc_train_pep) > 0, "Training set is empty"
assert len(mhc_train_pep) == len(mhc_train_lab), "Mismatch in training data lengths"
assert set(mhc_train_lab) <= {0, 1}, "Labels must be binary"
```

### 14. **Inefficient Embedding Extraction**
**Location**: Cell 1, lines 134-145  
**Issue**: Processes embeddings one at a time in loop  
**Impact**: Slow, could be batched  
**Fix**: Batch processing (though current approach may be necessary for memory constraints)

### 15. **Test Set Shuffling**
**Location**: Cell 1, line 157  
**Issue**: `shuffle=True` for test dataloader  
**Impact**: Unnecessary, test set should be deterministic  
**Fix**: `shuffle=False` for test/validation dataloaders

---

## 🔵 Minor Issues / Best Practices

### 16. **Missing Type Hints**
**Location**: All functions and classes  
**Issue**: No type annotations  
**Impact**: Less clear function signatures  
**Fix**: Add type hints for better IDE support and documentation

### 17. **Inconsistent Comments**
**Location**: Throughout  
**Issue**: Some code well-commented, other complex parts lack explanation  
**Impact**: Hard to understand complex logic (e.g., attention mask creation)  
**Fix**: Add docstrings to classes and complex functions

### 18. **Missing Error Handling**
**Location**: Model loading, file I/O  
**Issue**: No try/except for file operations  
**Impact**: Notebook crashes on missing files  
**Fix**: Add error handling:
```python
try:
    model.load_state_dict(torch.load("./attention_model.pth", weights_only=True))
except FileNotFoundError:
    print("Model file not found. Please train the model first.")
    raise
```

### 19. **Git Clone in Notebook**
**Location**: Cell 1, line 38  
**Issue**: `!git clone` runs every time cell executes  
**Impact**: Unnecessary, should check if directory exists  
**Fix**:
```python
import os
if not os.path.exists('./BioLLMComposition'):
    !git clone https://github.com/jjoecclark/BioLLMComposition
```

### 20. **Redundant Array Conversion**
**Location**: Multiple visualization cells  
**Issue**: `embs_cat = np.array(embs_cat)` when already an array  
**Impact**: Minor inefficiency  
**Fix**: Remove redundant conversions

### 21. **Inconsistent EPOCHS Variable**
**Location**: Cell 11, line 936  
**Issue**: `EPOCHS=100` redefined in attention model cell  
**Impact**: Could cause confusion if global EPOCHS changes  
**Fix**: Remove local redefinition, use global constant

### 22. **Missing Legend in Visualizations**
**Location**: All visualization cells  
**Issue**: Legend entries created but `plt.legend()` never called  
**Impact**: Legends don't appear in plots  
**Fix**: Add `plt.legend()` before `plt.savefig()`

### 23. **Dataset Class Typo**
**Location**: Cell 1, line 116  
**Issue**: Comment says "Peptide sequenes" (typo: "sequences")  
**Impact**: Minor documentation issue  
**Fix**: Correct spelling

### 24. **Inconsistent Device Management**
**Location**: Throughout  
**Issue**: Some tensors moved to device multiple times, some not moved when needed  
**Impact**: Potential device mismatch errors  
**Fix**: Ensure all tensors are on correct device before operations

---

## 📊 Performance Considerations

### 25. **Tokenization on Device**
**Location**: Cell 1, lines 148-151  
**Issue**: Tokenized data moved to device, but tokens are dictionaries  
**Impact**: Only tensor values moved, not the dict structure  
**Fix**: Move individual tensor values or keep on CPU until needed

### 26. **Gradient Computation for Frozen Models**
**Location**: Attention and Composition models  
**Issue**: `with torch.no_grad()` used correctly, but could be optimized further  
**Impact**: Minor - current implementation is fine

### 27. **Batch Size Hard-coded**
**Location**: Cell 1, lines 155, 157  
**Issue**: Batch size 128 hard-coded  
**Impact**: May not be optimal for all models  
**Fix**: Make configurable or adjust per model

---

## ✅ Positive Aspects

1. **Good use of `torch.no_grad()`** for inference and frozen models
2. **Proper model evaluation mode** (`model.eval()`) during validation
3. **Clear separation** of different model architectures
4. **Good use of tqdm** for progress tracking
5. **Proper device management** in most places
6. **Model state saving** for best checkpoints (mostly)

---

## 🔧 Recommended Refactoring

1. **Extract common training loop** into reusable function
2. **Create configuration dictionary** for all hyperparameters
3. **Add logging** instead of print statements
4. **Create utility functions** for visualization
5. **Add unit tests** for data preprocessing and model components
6. **Use dataclasses** for model configurations
7. **Add early stopping** to prevent overfitting
8. **Implement learning rate scheduling** (ExponentialLR imported but not used)

---

## 📝 Summary

**Total Issues Found**: 27
- **Critical**: 4
- **Major**: 5  
- **Moderate**: 6
- **Minor**: 12

**Priority Actions**:
1. Add random seeds for reproducibility
2. Fix attention mask in CompositionModel
3. Fix data loss in preprocessing
4. Standardize model saving
5. Fix visualization issues
6. Extract common training code

The code is functional but needs refactoring for maintainability, reproducibility, and correctness. The most critical issues should be addressed before publication or further experimentation.

