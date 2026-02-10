"""
Protein-DNA LLM Composition Demo

This script trains multiple models for protein-DNA binding prediction using:
- DNABERT-2 for DNA sequence embeddings (768-dim)
- ESM2 for protein sequence embeddings (320-dim)

Data: DNAProDB2 protein-DNA chain-wise interactions with cluster-based splitting
to prevent data leakage (proteins clustered at 30% sequence identity).
"""

import math
import torch
import random
import matplotlib
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel, BertConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Hyperparameters
LR = 5e-4
EPOCHS = 100
VERBOSE = False
BATCHSIZE = 16

# Embedding dimensions
DNA_EMB_DIM = 768   # DNABERT-2 output dimension
PROT_EMB_DIM = 320  # ESM2 output dimension

# ==============================================================================
# Data Loading and Splitting
# ==============================================================================

DATA_PATH = "/home/zcorn/Projects/proteinDNA_data/working/dnaprodb2/dna_protein_chain_wise_cleaned.parquet"

print("Loading dataset...")
df = pd.read_parquet(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Label distribution:\n{df['label'].value_counts()}")
print(f"Number of unique clusters: {df['cluster_id'].nunique()}")

# Cluster-based train/test split to prevent data leakage
# Split by cluster_id so that proteins from the same cluster don't appear in both train and test
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, val_idx = next(gss.split(df, groups=df['cluster_id']))

train_df = df.iloc[train_idx].reset_index(drop=True)
val_df = df.iloc[val_idx].reset_index(drop=True)

print(f"\nTrain size: {len(train_df)}, Val size: {len(val_df)}")
print(f"Train clusters: {train_df['cluster_id'].nunique()}, Val clusters: {val_df['cluster_id'].nunique()}")

# Verify no cluster overlap
train_clusters = set(train_df['cluster_id'].unique())
val_clusters = set(val_df['cluster_id'].unique())
assert len(train_clusters & val_clusters) == 0, "Cluster overlap detected between train and val!"
print("No cluster overlap between train and val sets.")

# Extract sequences and labels
train_dna_seqs = train_df['dna_seq'].tolist()
train_prot_seqs = train_df['prot_seq'].tolist()
train_labels = train_df['label'].tolist()

val_dna_seqs = val_df['dna_seq'].tolist()
val_prot_seqs = val_df['prot_seq'].tolist()
val_labels = val_df['label'].tolist()

# ==============================================================================
# Load Language Models
# ==============================================================================

print("\nLoading DNABERT-2 model...")
# DNA Language Model (DNABERT-2)
dna_tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
dna_config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
dna_lm = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=dna_config)
dna_lm = dna_lm.to(device).eval()

print("Loading ESM2 model...")
# Protein Language Model (ESM2)
esm_layers = 6
esm_params = 8
prot_lm = AutoModelForMaskedLM.from_pretrained(f'facebook/esm2_t{esm_layers}_{esm_params}M_UR50D').to(device).eval()
prot_tokenizer = AutoTokenizer.from_pretrained(f'facebook/esm2_t{esm_layers}_{esm_params}M_UR50D')

print("Models loaded successfully!")


# Get the mean embedding from DNABERT-2
def get_dna_mean_rep(sequence):
    """Get mean pooled embedding from DNABERT-2."""
    inputs = dna_tokenizer(sequence, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = dna_lm(**inputs)
    # DNABERT-2 returns tuple: (last_hidden_state, pooler_output) or just (last_hidden_state,)
    # Access first element which is the hidden states
    if isinstance(outputs, tuple):
        hidden_states = outputs[0][0]  # [seq_len, 768]
    else:
        hidden_states = outputs.last_hidden_state[0]  # [seq_len, 768]
    # Mean pooling over all tokens
    mean_embedding = hidden_states.mean(dim=0)
    return mean_embedding.cpu().numpy()


# Get the mean embedding from ESM2
def get_prot_mean_rep(sequence):
    """Get mean pooled embedding from ESM2."""
    token_ids = prot_tokenizer(sequence, return_tensors='pt', truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        results = prot_lm.forward(token_ids.input_ids, output_hidden_states=True)
    representations = results.hidden_states[-1][0]
    # Exclude BOS/EOS tokens: [1:len(sequence)+1]
    mean_embedding = representations[1:min(len(sequence)+1, representations.shape[0]-1)].mean(dim=0)
    return mean_embedding.cpu().numpy()


# Custom data set class for DNA-protein pairs
class DNAProteinDataset(Dataset):
    def __init__(self, dna_embs, protein_embs, labels, dna_tokens, prot_tokens):
        self.dna_embs = dna_embs        # DNA embeddings (precomputed)
        self.protein_embs = protein_embs  # Protein embeddings (precomputed)
        self.labels = labels              # Binary labels
        self.dna_tokens = dna_tokens      # DNABERT tokens for the sequences
        self.prot_tokens = prot_tokens    # ESM tokens for the sequences

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, i):
        dna = self.dna_embs[i]
        prot = self.protein_embs[i]
        y = self.labels[i]
        dna_tok = {key: value[i] for key, value in self.dna_tokens.items()}
        prot_tok = {key: value[i] for key, value in self.prot_tokens.items()}
        return dna, prot, y, dna_tok, prot_tok


# ==============================================================================
# Extract Embeddings
# ==============================================================================

print("\nExtracting training embeddings...")
dna_embs_train, protein_embs_train = [], []
for dna_seq, prot_seq in tqdm(zip(train_dna_seqs, train_prot_seqs), total=len(train_dna_seqs)):
    dna_embs_train.append(get_dna_mean_rep(dna_seq))
    protein_embs_train.append(get_prot_mean_rep(prot_seq))
dna_embs_train, protein_embs_train = np.array(dna_embs_train), np.array(protein_embs_train)

print("Extracting validation embeddings...")
dna_embs_val, protein_embs_val = [], []
for dna_seq, prot_seq in tqdm(zip(val_dna_seqs, val_prot_seqs), total=len(val_dna_seqs)):
    dna_embs_val.append(get_dna_mean_rep(dna_seq))
    protein_embs_val.append(get_prot_mean_rep(prot_seq))
dna_embs_val, protein_embs_val = np.array(dna_embs_val), np.array(protein_embs_val)

# ==============================================================================
# Tokenize for Attention/Composition Models
# ==============================================================================

print("\nTokenizing sequences...")
# Determine max lengths based on data
max_dna_len = min(512, max(len(s) for s in train_dna_seqs + val_dna_seqs) + 10)
max_prot_len = min(1024, max(len(s) for s in train_prot_seqs + val_prot_seqs) + 10)
print(f"Max DNA length: {max_dna_len}, Max Protein length: {max_prot_len}")

dna_tokens_train = dna_tokenizer(train_dna_seqs, return_tensors='pt', padding='max_length', 
                                  max_length=max_dna_len, truncation=True).to(device)
prot_tokens_train = prot_tokenizer(train_prot_seqs, return_tensors='pt', padding='max_length', 
                                    max_length=max_prot_len, truncation=True).to(device)
dna_tokens_val = dna_tokenizer(val_dna_seqs, return_tensors='pt', padding='max_length', 
                                max_length=max_dna_len, truncation=True).to(device)
prot_tokens_val = prot_tokenizer(val_prot_seqs, return_tensors='pt', padding='max_length', 
                                  max_length=max_prot_len, truncation=True).to(device)

# Make data sets
train_dataset = DNAProteinDataset(dna_embs_train, protein_embs_train, np.array(train_labels), 
                                   dna_tokens_train, prot_tokens_train)
train_dataloader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)
test_dataset = DNAProteinDataset(dna_embs_val, protein_embs_val, np.array(val_labels), 
                                  dna_tokens_val, prot_tokens_val)
test_dataloader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=True)

# Initialize results storage with metrics
results = {
    'Protein only': {'accuracy': [], 'precision': [], 'recall': [], 'mcc': [], 'f1': [], 'roc_auc': [], 'pr_auc': []},
    'DNA only': {'accuracy': [], 'precision': [], 'recall': [], 'mcc': [], 'f1': [], 'roc_auc': [], 'pr_auc': []},
    'Concatenation': {'accuracy': [], 'precision': [], 'recall': [], 'mcc': [], 'f1': [], 'roc_auc': [], 'pr_auc': []},
    'Contrastive': {'accuracy': [], 'precision': [], 'recall': [], 'mcc': [], 'f1': [], 'roc_auc': [], 'pr_auc': []},
    'Attention': {'accuracy': [], 'precision': [], 'recall': [], 'mcc': [], 'f1': [], 'roc_auc': [], 'pr_auc': []},
    'Composition': {'accuracy': [], 'precision': [], 'recall': [], 'mcc': [], 'f1': [], 'roc_auc': [], 'pr_auc': []},
}


def compute_metrics(y_true, y_pred, y_prob):
    """
    Compute evaluation metrics: Accuracy, ROC-AUC, PR-AUC, MCC, Precision, Recall, F1.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_prob: Predicted probabilities for positive class
        
    Returns:
        dict with accuracy, roc_auc, pr_auc, mcc, precision, recall, f1
    """
    acc = accuracy_score(y_true, y_pred)
    
    # Handle edge case where only one class is present
    roc_auc = roc_auc_score(y_true, y_prob)
    
    pr_auc = average_precision_score(y_true, y_prob)
    
    mcc = matthews_corrcoef(y_true, y_pred)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        'accuracy': acc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'mcc': mcc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def evaluate_model(model, dataloader, model_type='standard'):
    """
    Evaluate model on dataloader and compute all metrics.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation
        model_type: 'protein_only', 'dna_only', 'concat', 'contrastive', 
                   'attention', or 'composition'
    
    Returns:
        dict with all metrics
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            dna_embs, protein_embs, labels, dna_tokens, prot_tokens = batch
            labels_np = labels.numpy().astype(int)
            
            if model_type == 'protein_only':
                protein_embs = protein_embs.to(device)
                outputs = model(protein_embs)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
            elif model_type == 'dna_only':
                dna_embs = dna_embs.to(device)
                outputs = model(dna_embs)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
            elif model_type == 'concat':
                inputs = torch.cat((dna_embs, protein_embs), dim=1).to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
            elif model_type == 'contrastive':
                dna_embs, protein_embs = dna_embs.to(device), protein_embs.to(device)
                dna_out, prot_out = model(dna_embs, protein_embs)
                similarities = torch.nn.functional.cosine_similarity(dna_out, prot_out, dim=1)
                # Map similarities [-1, 1] to probabilities [0, 1] using (sim + 1) / 2
                # This is more appropriate than sigmoid for cosine similarity
                probs = ((similarities + 1) / 2).cpu().numpy()
                # Predict positive (1) if similarity > 0, else negative (0)
                preds = (similarities > 0).long().cpu().numpy()
                
            elif model_type in ['attention', 'composition']:
                outputs, _ = model(dna_tokens, prot_tokens)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            
            all_labels.extend(labels_np)
            all_preds.extend(preds)
            all_probs.extend(probs)
    
    all_labels = np.array(all_labels, dtype=int)
    all_preds = np.array(all_preds, dtype=int)
    all_probs = np.array(all_probs, dtype=float)
    
    return compute_metrics(all_labels, all_preds, all_probs)

# ==============================================================================
# Train model on protein embeddings only
# ==============================================================================
print("\n" + "="*60)
print("Training: Protein only baseline")
print("="*60)

for n in range(3):
    model = torch.nn.Sequential(
        torch.nn.Linear(PROT_EMB_DIM, 2),
    )
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_metrics = None
    best_roc_auc = -1  # Start at -1 to ensure first epoch always updates
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_dataloader:
            _, protein_embs, labels, _, _ = batch
            protein_embs, labels = protein_embs.to(device), labels.to(device)
            outputs = model(protein_embs)
            loss = criterion(outputs, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_loss /= total
        train_accuracy = correct / total

        # Evaluate with all metrics
        metrics = evaluate_model(model, test_dataloader, model_type='protein_only')
        
        if metrics['roc_auc'] > best_roc_auc:
            best_roc_auc = metrics['roc_auc']
            best_metrics = metrics
            torch.save(model.state_dict(), "./results/protein_emb_model.pth")
        if VERBOSE: 
            print(f"Epoch {epoch + 1}, Train Acc: {train_accuracy:.4f}, "
                  f"Test Acc: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")

    print(f"Run {n+1}: Acc={best_metrics['accuracy']:.4f}, Prec={best_metrics['precision']:.4f}, "
          f"Rec={best_metrics['recall']:.4f}, MCC={best_metrics['mcc']:.4f}, F1={best_metrics['f1']:.4f}, "
          f"ROC-AUC={best_metrics['roc_auc']:.4f}, PR-AUC={best_metrics['pr_auc']:.4f}")
    for metric_name in ['accuracy', 'precision', 'recall', 'mcc', 'f1', 'roc_auc', 'pr_auc']:
        results['Protein only'][metric_name].append(best_metrics[metric_name])

# ==============================================================================
# Train model on DNA embeddings only
# ==============================================================================
print("\n" + "="*60)
print("Training: DNA only baseline")
print("="*60)

for n in range(3):
    model = torch.nn.Sequential(
        torch.nn.Linear(DNA_EMB_DIM, 2),
    )
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_metrics = None
    best_roc_auc = -1  # Start at -1 to ensure first epoch always updates
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_dataloader:
            dna_embs, _, labels, _, _ = batch
            dna_embs, labels = dna_embs.to(device), labels.to(device)
            outputs = model(dna_embs)
            loss = criterion(outputs, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_loss /= total
        train_accuracy = correct / total

        # Evaluate with all metrics
        metrics = evaluate_model(model, test_dataloader, model_type='dna_only')
        
        if metrics['roc_auc'] > best_roc_auc:
            best_roc_auc = metrics['roc_auc']
            best_metrics = metrics
            torch.save(model.state_dict(), "./results/dna_emb_model.pth")
        if VERBOSE: 
            print(f"Epoch {epoch + 1}, Train Acc: {train_accuracy:.4f}, "
                  f"Test Acc: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")

    print(f"Run {n+1}: Acc={best_metrics['accuracy']:.4f}, Prec={best_metrics['precision']:.4f}, "
          f"Rec={best_metrics['recall']:.4f}, MCC={best_metrics['mcc']:.4f}, F1={best_metrics['f1']:.4f}, "
          f"ROC-AUC={best_metrics['roc_auc']:.4f}, PR-AUC={best_metrics['pr_auc']:.4f}")
    for metric_name in ['accuracy', 'precision', 'recall', 'mcc', 'f1', 'roc_auc', 'pr_auc']:
        results['DNA only'][metric_name].append(best_metrics[metric_name])

# ==============================================================================
# Train model on concatenated embeddings
# ==============================================================================
print("\n" + "="*60)
print("Training: Concatenation")
print("="*60)

for n in range(3):
    # DNA (768) + Protein (320) = 1088
    model = torch.nn.Sequential(
        torch.nn.Linear(DNA_EMB_DIM + PROT_EMB_DIM, 2),
    )
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_metrics = None
    best_roc_auc = -1  # Start at -1 to ensure first epoch always updates
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_dataloader:
            dna_embs, protein_embs, labels, _, _ = batch
            inputs = torch.cat((dna_embs, protein_embs), dim=1)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_loss /= total
        train_accuracy = correct / total

        # Evaluate with all metrics
        metrics = evaluate_model(model, test_dataloader, model_type='concat')
        
        if metrics['roc_auc'] > best_roc_auc:
            best_roc_auc = metrics['roc_auc']
            best_metrics = metrics
            torch.save(model.state_dict(), "./results/concat_model.pth")
        if VERBOSE: 
            print(f"Epoch {epoch + 1}, Train Acc: {train_accuracy:.4f}, "
                  f"Test Acc: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")

    print(f"Run {n+1}: Acc={best_metrics['accuracy']:.4f}, Prec={best_metrics['precision']:.4f}, "
          f"Rec={best_metrics['recall']:.4f}, MCC={best_metrics['mcc']:.4f}, F1={best_metrics['f1']:.4f}, "
          f"ROC-AUC={best_metrics['roc_auc']:.4f}, PR-AUC={best_metrics['pr_auc']:.4f}")
    for metric_name in ['accuracy', 'precision', 'recall', 'mcc', 'f1', 'roc_auc', 'pr_auc']:
        results['Concatenation'][metric_name].append(best_metrics[metric_name])

# ==============================================================================
# Contrastive Learning Model
# ==============================================================================
print("\n" + "="*60)
print("Training: Contrastive")
print("="*60)

import torch.nn.functional as F
from torch.nn import CosineSimilarity


class CLModel(nn.Module):
    def __init__(self):
        super(CLModel, self).__init__()
        # Projection network for DNA (768 -> 128)
        self.dna_proj = torch.nn.Sequential(
            torch.nn.Linear(DNA_EMB_DIM, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
        )
        # Projection network for protein (320 -> 128)
        self.protein_proj = torch.nn.Sequential(
            torch.nn.Linear(PROT_EMB_DIM, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
        )

    def forward(self, dna, prot):
        dna = self.dna_proj(dna)
        prot = self.protein_proj(prot)
        dna = F.normalize(dna, p=2, dim=1)
        prot = F.normalize(prot, p=2, dim=1)
        return dna, prot


for n in range(3):
    model = CLModel()
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CosineEmbeddingLoss()

    best_metrics = None
    best_roc_auc = -1  # Start at -1 to ensure first epoch always updates
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_dataloader:
            dna_embs, protein_embs, labels, _, _ = batch
            labels = labels.clone()
            labels[labels == 0] = -1

            dna_embs, protein_embs, labels = dna_embs.to(device), protein_embs.to(device), labels.to(device)
            dna_out, prot_out = model(dna_embs, protein_embs)
            loss = criterion(dna_out, prot_out, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss += loss.item() * labels.size(0)
            similarities = CosineSimilarity(dim=1, eps=1e-6)(dna_out, prot_out)
            preds = (similarities > 0).long() * 2 - 1
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss /= total
        train_accuracy = correct / total

        # Evaluate with all metrics
        metrics = evaluate_model(model, test_dataloader, model_type='contrastive')
        
        if metrics['roc_auc'] > best_roc_auc:
            best_roc_auc = metrics['roc_auc']
            best_metrics = metrics
            torch.save(model.state_dict(), "./results/contrastive_model.pth")
        if VERBOSE: 
            print(f"Epoch {epoch + 1}, Train Acc: {train_accuracy:.4f}, "
                  f"Test Acc: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")

    print(f"Run {n+1}: Acc={best_metrics['accuracy']:.4f}, Prec={best_metrics['precision']:.4f}, "
          f"Rec={best_metrics['recall']:.4f}, MCC={best_metrics['mcc']:.4f}, F1={best_metrics['f1']:.4f}, "
          f"ROC-AUC={best_metrics['roc_auc']:.4f}, PR-AUC={best_metrics['pr_auc']:.4f}")
    for metric_name in ['accuracy', 'precision', 'recall', 'mcc', 'f1', 'roc_auc', 'pr_auc']:
        results['Contrastive'][metric_name].append(best_metrics[metric_name])

# ==============================================================================
# Attention Model
# ==============================================================================
print("\n" + "="*60)
print("Training: Attention")
print("="*60)


class AttentionModel(nn.Module):
    """
    Cross-attention model: Protein attends to DNA.
    
    DNA (DNABERT-2): 768-dim embeddings
    Protein (ESM2): 320-dim embeddings
    
    We project DNA to protein dimension (320) for attention,
    then use a prediction head.
    """
    def __init__(self, dna_model, prot_model, dna_tok, prot_tok):
        super(AttentionModel, self).__init__()
        self.dna_model = dna_model
        self.prot_model = prot_model
        self.dna_tokenizer = dna_tok
        self.prot_tokenizer = prot_tok
        
        # Project DNA from 768 to 320 to match protein dimension
        self.dna_proj = nn.Linear(DNA_EMB_DIM, PROT_EMB_DIM)
        
        self.attn = nn.MultiheadAttention(PROT_EMB_DIM, 1, batch_first=True)
        self.prediction_head = torch.nn.Sequential(
            torch.nn.Linear(PROT_EMB_DIM, 2),
        )
        self.post_attn_norm = nn.LayerNorm(PROT_EMB_DIM)
        
        # Freeze the language models
        for param in self.dna_model.parameters():
            param.requires_grad = False
        for param in self.prot_model.parameters():
            param.requires_grad = False

    def forward(self, dna_tokens, prot_tokens, attn_mask=None):
        # Get DNA embeddings from DNABERT-2
        with torch.no_grad():
            dna_outputs = self.dna_model(
                input_ids=dna_tokens['input_ids'],
                attention_mask=dna_tokens['attention_mask']
            )
        # Handle tuple output from DNABERT-2
        if isinstance(dna_outputs, tuple):
            dna_hidden = dna_outputs[0]  # [B, L_dna, 768]
        else:
            dna_hidden = dna_outputs.last_hidden_state  # [B, L_dna, 768]
        
        # Project DNA to protein dimension
        dna_hidden = self.dna_proj(dna_hidden)  # [B, L_dna, 320]
        
        # Get protein embeddings from ESM2
        with torch.no_grad():
            prot_outputs = self.prot_model.forward(
                prot_tokens['input_ids'],
                prot_tokens['attention_mask'],
                output_hidden_states=True
            )
        prot_hidden = prot_outputs.hidden_states[-1]  # [B, L_prot, 320]
        
        # Create attention mask
        attn_mask = torch.matmul(
            prot_tokens["attention_mask"].unsqueeze(2).float(),
            dna_tokens['attention_mask'].unsqueeze(1).float(),
        ).repeat(1, 1, 1)
        
        # Perform cross-attention: protein (query) attends to DNA (key, value)
        output, attn_weights = self.attn(
            query=prot_hidden, key=dna_hidden, value=dna_hidden,
            attn_mask=attn_mask, average_attn_weights=False
        )
        output = self.post_attn_norm(output) + prot_hidden
        
        # Mean pooling over protein positions
        mask_sum = prot_tokens['attention_mask'].sum(dim=1, keepdim=True).clamp(min=1e-6)
        output = (output * prot_tokens['attention_mask'].unsqueeze(2)).sum(dim=1) / mask_sum
        
        return self.prediction_head(output), output


for n in range(3):
    model = AttentionModel(dna_lm, prot_lm, dna_tokenizer, prot_tokenizer)
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_metrics = None
    best_roc_auc = -1  # Start at -1 to ensure first epoch always updates
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_dataloader:
            _, _, labels, dna_tokens, prot_tokens = batch
            labels = labels.to(device)
            outputs, _ = model(dna_tokens, prot_tokens)
            loss = criterion(outputs, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_loss /= total
        train_accuracy = correct / total

        # Evaluate with all metrics
        metrics = evaluate_model(model, test_dataloader, model_type='attention')
        
        if metrics['roc_auc'] > best_roc_auc:
            best_roc_auc = metrics['roc_auc']
            best_metrics = metrics
            torch.save(model.state_dict(), "./results/attention_model.pth")
        if VERBOSE: 
            print(f"Epoch {epoch + 1}, Train Acc: {train_accuracy:.4f}, "
                  f"Test Acc: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    print(f"Run {n+1}: Acc={best_metrics['accuracy']:.4f}, Prec={best_metrics['precision']:.4f}, "
          f"Rec={best_metrics['recall']:.4f}, MCC={best_metrics['mcc']:.4f}, F1={best_metrics['f1']:.4f}, "
          f"ROC-AUC={best_metrics['roc_auc']:.4f}, PR-AUC={best_metrics['pr_auc']:.4f}")
    for metric_name in ['accuracy', 'precision', 'recall', 'mcc', 'f1', 'roc_auc', 'pr_auc']:
        results['Attention'][metric_name].append(best_metrics[metric_name])

# ==============================================================================
# Composition Model
# ==============================================================================
print("\n" + "="*60)
print("Training: Composition")
print("="*60)

# Target layers for cross-attention in ESM2
target_layers = [0, 3, 5]


class CompositionModel(nn.Module):
    """
    Composition of Language Models for Protein-DNA Binding.
    
    Injects DNA information into protein transformer layers via cross-attention.
    DNA embeddings are projected to protein dimension before cross-attention.
    """
    def __init__(self, dna_model, prot_model, dna_tok, prot_tok):
        super(CompositionModel, self).__init__()
        self.dna_model = dna_model
        self.prot_model = prot_model
        self.dna_tokenizer = dna_tok
        self.prot_tokenizer = prot_tok
        
        # Project DNA from 768 to 320 to match protein dimension
        self.dna_proj = nn.Linear(DNA_EMB_DIM, PROT_EMB_DIM)
        
        self.cross_atten_layers = nn.ModuleList([
            nn.MultiheadAttention(PROT_EMB_DIM, 20, batch_first=True) for _ in range(len(target_layers))
        ])
        self.post_attn_norms = nn.ModuleList([
            nn.LayerNorm(PROT_EMB_DIM) for _ in range(len(target_layers))
        ])
        self.prediction_head = torch.nn.Sequential(
            torch.nn.Linear(PROT_EMB_DIM, 2),
        )
        self.esm_layers = 6
        
        # Freeze the language models
        for param in self.dna_model.parameters():
            param.requires_grad = False
        for param in self.prot_model.parameters():
            param.requires_grad = False

    def forward(self, dna_input, prot_input, attn_mask=None):
        # Get DNA embeddings from DNABERT-2
        with torch.no_grad():
            dna_outputs = self.dna_model(
                input_ids=dna_input['input_ids'],
                attention_mask=dna_input['attention_mask']
            )
        # Handle tuple output from DNABERT-2
        if isinstance(dna_outputs, tuple):
            dna = dna_outputs[0]  # [B, L_dna, 768]
        else:
            dna = dna_outputs.last_hidden_state  # [B, L_dna, 768]
        
        # Project DNA to protein dimension
        dna = self.dna_proj(dna)  # [B, L_dna, 320]
        
        # Create cross-attention mask
        attn_mask = torch.matmul(
            prot_input["attention_mask"].unsqueeze(2).float(),
            dna_input['attention_mask'].unsqueeze(1).float(),
        ).repeat(20, 1, 1)
        
        # Create the ESM attention masks correctly
        prot_attn_mask = self.prot_model.get_extended_attention_mask(
            prot_input['attention_mask'], prot_input["input_ids"].size()
        )
        
        # Embedding layer for protein
        prot = self.prot_model.esm.embeddings(
            prot_input["input_ids"], prot_input["attention_mask"]
        ).to(device)
        
        # Layerwise forward pass with cross-attention injection
        counter = 0
        for i in range(0, self.esm_layers):
            # Update protein embeddings through ESM layer
            prot = self.prot_model.esm.encoder.layer[i](prot, prot_attn_mask)[0]
            
            if i in target_layers:
                # Perform cross attention: protein attends to DNA
                attn_out, _ = self.cross_atten_layers[counter](
                    query=prot, key=dna, value=dna,
                    attn_mask=attn_mask, average_attn_weights=False
                )
                attn_out = self.post_attn_norms[counter](attn_out)
                prot = prot + attn_out
                counter += 1
        
        prot = self.prot_model.esm.encoder.emb_layer_norm_after(prot)
        
        # Mean pooling over protein positions
        mask_sum = prot_input['attention_mask'].sum(dim=1, keepdim=True).clamp(min=1e-6)
        prot = (prot * prot_input['attention_mask'].unsqueeze(2)).sum(dim=1) / mask_sum
        
        return self.prediction_head(prot), prot


for n in range(3):
    model = CompositionModel(dna_lm, prot_lm, dna_tokenizer, prot_tokenizer)
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_metrics = None
    best_roc_auc = -1  # Start at -1 to ensure first epoch always updates
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_dataloader:
            _, _, labels, dna_tokens, prot_tokens = batch
            labels = labels.to(device)
            outputs, _ = model(dna_tokens, prot_tokens)
            loss = criterion(outputs, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_loss /= total
        train_accuracy = correct / total

        # Evaluate with all metrics
        metrics = evaluate_model(model, test_dataloader, model_type='composition')
        
        if metrics['roc_auc'] > best_roc_auc:
            best_roc_auc = metrics['roc_auc']
            best_metrics = metrics
            torch.save(model.state_dict(), "./results/comp_model.pth")
        if VERBOSE: 
            print(f"Epoch {epoch + 1}, Train Acc: {train_accuracy:.4f}, "
                  f"Test Acc: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")

    print(f"Run {n+1}: Acc={best_metrics['accuracy']:.4f}, Prec={best_metrics['precision']:.4f}, "
          f"Rec={best_metrics['recall']:.4f}, MCC={best_metrics['mcc']:.4f}, F1={best_metrics['f1']:.4f}, "
          f"ROC-AUC={best_metrics['roc_auc']:.4f}, PR-AUC={best_metrics['pr_auc']:.4f}")
    for metric_name in ['accuracy', 'precision', 'recall', 'mcc', 'f1', 'roc_auc', 'pr_auc']:
        results['Composition'][metric_name].append(best_metrics[metric_name])

# ==============================================================================
# Save Results
# ==============================================================================

# Create comprehensive results dataframe with all metrics
rows = []
for model_name, metrics_dict in results.items():
    for metric_name, values in metrics_dict.items():
        row = {
            'Model': model_name,
            'Metric': metric_name,
            'Run 1': values[0] if len(values) > 0 else None,
            'Run 2': values[1] if len(values) > 1 else None,
            'Run 3': values[2] if len(values) > 2 else None,
        }
        if len(values) >= 3:
            row['Mean'] = np.mean(values)
            row['Std'] = np.std(values)
        rows.append(row)

df_results = pd.DataFrame(rows)

print("\n" + "="*60)
print("Final Results")
print("="*60)
print(df_results.to_string(index=False))

# Save detailed results
df_results.to_csv('./results/results_tannerHP_ProteinDNA_20260201.csv', index=False)
print("\nResults saved to ./results/results_tannerHP_ProteinDNA_20260201.csv")

# Also create a summary table (mean ± std for each metric)
print("\n" + "="*60)
print("Summary Table (Mean ± Std)")
print("="*60)

summary_rows = []
for model_name in results.keys():
    row = {'Model': model_name}
    for metric_name in ['accuracy', 'precision', 'recall', 'mcc', 'f1', 'roc_auc', 'pr_auc']:
        values = results[model_name][metric_name]
        if len(values) >= 3:
            row[metric_name] = f"{np.mean(values):.4f} ± {np.std(values):.4f}"
        else:
            row[metric_name] = "N/A"
    summary_rows.append(row)

df_summary = pd.DataFrame(summary_rows)
print(df_summary.to_string(index=False))
df_summary.to_csv('./results/results_tannerHP_ProteinDNA_summary_20260201.csv', index=False)
print("\nSummary saved to ./results/results_tannerHP_ProteinDNA_summary_20260201.csv")
