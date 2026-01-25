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
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from transformers import AutoModelForMaskedLM, AutoTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Load MHC data set
MHCtrain = pd.read_csv('./data/combo_1and2_train.tsv', sep='\t')
MHCval = pd.read_csv('./data/combo_1and2_valid.tsv', sep='\t')

# Extract sequences from data frames
train_seqs = MHCtrain['target_chainseq'].tolist()
train_labs = MHCtrain['binder'].tolist()
val_seqs = MHCval['target_chainseq'].tolist()
val_labs = MHCval['binder'].tolist()

# Preprocess sequences (protein and peptide are separated by '/')
mhc_train_pep = []
mhc_train_rec = []
mhc_train_lab = []
for i, s in enumerate(train_seqs):
    try:
        rec, pep = s.split('/')
        mhc_train_pep.append(pep)
        mhc_train_rec.append(rec)
        mhc_train_lab.append(train_labs[i])
    except:
        pass

mhc_val_pep = []
mhc_val_rec = []
mhc_val_lab = []
for i, s in enumerate(val_seqs):
    try:
        rec, pep = s.split('/')
        mhc_val_pep.append(pep)
        mhc_val_rec.append(rec)
        mhc_val_lab.append(val_labs[i])
    except:
        pass

# Load pLM and PLM models
esm_layers = 6
esm_params = 8
plm = AutoModelForMaskedLM.from_pretrained(f'facebook/esm2_t{esm_layers}_{esm_params}M_UR50D').to(device).eval()
PLM = AutoModelForMaskedLM.from_pretrained(f'facebook/esm2_t{esm_layers}_{esm_params}M_UR50D').to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(f'facebook/esm2_t{esm_layers}_{esm_params}M_UR50D')

# Get the mean embedding from esm style model
def get_mean_rep(model_name, sequence):
    token_ids = tokenizer(sequence, return_tensors='pt').to(device)
    with torch.no_grad():
        results = model_name.forward(token_ids.input_ids, output_hidden_states=True)
    representations = results.hidden_states[-1][0]
    mean_embedding = representations[1:len(sequence)+1].mean(dim=0)
    return mean_embedding.cpu().numpy()

# Custom data set class for peptide-protein pairs
class mhcdataset(Dataset):
    def __init__(self, peptides, proteins, labels, p_tokens, P_tokens):
        self.peptides = peptides # Peptide sequenes
        self.proteins = proteins # Protein sequences
        self.labels = labels # Binary labels
        self.p_tokens = p_tokens # ESM tokens for the sequences
        self.P_tokens = P_tokens

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, i):
        p = self.peptides[i]
        P = self.proteins[i]
        y = self.labels[i]
        p_tokens = {key: value[i] for key, value in self.p_tokens.items()}
        P_tokens = {key: value[i] for key, value in self.P_tokens.items()}
        return p, P, y, p_tokens, P_tokens

# Extract training data set embeddings
peptide_embs_train, protein_embs_train = [], []
for pep, pro in tqdm(zip(mhc_train_pep, mhc_train_rec)):
    peptide_embs_train.append(get_mean_rep(plm, pep))
    protein_embs_train.append(get_mean_rep(PLM, pro))
peptide_embs_train, protein_embs_train = np.array(peptide_embs_train), np.array(protein_embs_train)

# Extract test data set embeddings
peptide_embs_val, protein_embs_val = [], []
for pep, pro in tqdm(zip(mhc_val_pep, mhc_val_rec)):
    peptide_embs_val.append(get_mean_rep(plm, pep))
    protein_embs_val.append(get_mean_rep(PLM, pro))
peptide_embs_val, protein_embs_val = np.array(peptide_embs_val), np.array(protein_embs_val)

# Tokenize
pep_tokens_train = tokenizer(mhc_train_pep, return_tensors='pt', padding='max_length', max_length=9, truncation=True).to(device)
pro_tokens_train = tokenizer(mhc_train_rec, return_tensors='pt', padding='max_length', max_length=181, truncation=True).to(device)
pep_tokens_val = tokenizer(mhc_val_pep, return_tensors='pt', padding='max_length', max_length=9, truncation=True).to(device)
pro_tokens_val = tokenizer(mhc_val_rec, return_tensors='pt', padding='max_length', max_length=181, truncation=True).to(device)

# Make data sets
train_data_set = mhcdataset(peptide_embs_train, protein_embs_train, np.array(mhc_train_lab), pep_tokens_train, pro_tokens_train)
train_dataloader = DataLoader(train_data_set, batch_size=BATCHSIZE, shuffle=True)
test_data_set = mhcdataset(peptide_embs_val, protein_embs_val, np.array(mhc_val_lab), pep_tokens_val, pro_tokens_val)
test_dataloader = DataLoader(test_data_set, batch_size=BATCHSIZE, shuffle=True)

# Initialize results storage
results = {
    'Protein only': [],
    'Peptide only': [],
    'Concatenation': [],
    'Contrastive': [],
    'Attention': [],
    'Composition': []
}

#@title Train model on protein embeddings only

# Train the model 3 independent times. Report the highest performance on the test set each time.
for n in range(3):

    # Simple model trained on concatenated peptide-protein embeddings
    model = torch.nn.Sequential(
        torch.nn.Linear(320, 2),
        # torch.nn.ReLU(),
        # torch.nn.Linear(128, 2),
        # torch.nn.Softmax(dim=1)
    )

    # Move to device, define optimizer, loss
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Record the best test set accuracy
    best_test_acc = 0

    # 500 training epochs per run
    for epoch in tqdm(range(EPOCHS)):

        # Start training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # Load data for batch
        for batch in train_dataloader:
            _, protein_embs, labels, _, _ = batch

            # perform forward pass, compute loss
            protein_embs, labels = protein_embs.to(device), labels.to(device)
            outputs = model(protein_embs)
            loss = criterion(outputs, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Get classification performance
            train_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_loss /= total
        train_accuracy = correct / total

        # Evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_dataloader:
                _, protein_embs, labels, _, _ = batch

                # Concatenate, forward pass, compute loss
                protein_embs, labels = protein_embs.to(device), labels.to(device)
                outputs = model(protein_embs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * labels.size(0)

                # Get classification performance
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            test_loss /= total
            test_accuracy = correct / total

            # Save if model had best accuracy so far
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                torch.save(model.state_dict(), "protein_emb_model.pth")
            if VERBOSE: print(f"Epoch {epoch + 1}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Print the best performance over all 500 epochs
    print(best_test_acc)
    results['Protein only'].append(best_test_acc)

#@title Train model on peptide embeddings only

# Train the model 3 independent times. Report the highest performance on the test set each time.
for n in range(3):

    # Simple model trained on concatenated peptide-protein embeddings
    model = torch.nn.Sequential(
        torch.nn.Linear(320, 2),
        # torch.nn.ReLU(),
        # torch.nn.Linear(128, 2),
        # torch.nn.Softmax(dim=1)
    )

    # Move to device, define optimizer, loss
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Record the best test set accuracy
    best_test_acc = 0

    # 500 training epochs per run
    for epoch in tqdm(range(EPOCHS)):

        # Start training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # Load data for batch
        for batch in train_dataloader:
            peptide_embs, _, labels, _, _ = batch

            # perform forward pass, compute loss
            peptide_embs, labels = peptide_embs.to(device), labels.to(device)
            outputs = model(peptide_embs)
            loss = criterion(outputs, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Get classification performance
            train_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_loss /= total
        train_accuracy = correct / total

        # Evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_dataloader:
                peptide_embs, _, labels, _, _ = batch

                # forward pass, compute loss
                peptide_embs, labels = peptide_embs.to(device), labels.to(device)
                outputs = model(peptide_embs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * labels.size(0)

                # Get classification performance
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            test_loss /= total
            test_accuracy = correct / total

            # Save if model had best accuracy so far
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                torch.save(model.state_dict(), "peptide_emb_model.pth")
            if VERBOSE: print(f"Epoch {epoch + 1}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Print the best performance over all 500 epochs
    print(best_test_acc)
    results['Peptide only'].append(best_test_acc)

#@title Train model on concatenated embeddings
# Train the model 3 independent times. Report the highest performance on the test set each time.
for n in range(3):

    # Simple model trained on concatenated peptide-protein embeddings
    model = torch.nn.Sequential(
        torch.nn.Linear(640, 2),
        # torch.nn.ReLU(),
        # torch.nn.Linear(128, 2),
        # torch.nn.Softmax(dim=1)
    )

    # Move to device, define optimizer, loss
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Record the best test set accuracy
    best_test_acc = 0

    # 500 training epochs per run
    for epoch in tqdm(range(EPOCHS)):

        # Start training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # Load data for batch
        for batch in train_dataloader:
            peptide_embs, protein_embs, labels, _, _ = batch

            # Concatenation occurs here
            inputs = torch.cat((peptide_embs, protein_embs), dim=1)

            # perform forward pass, compute loss
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Get classification performance
            train_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_loss /= total
        train_accuracy = correct / total

        # Evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_dataloader:
                peptide_embs, protein_embs, labels, _, _ = batch

                # Concatenate, forward pass, compute loss
                inputs = torch.cat((peptide_embs, protein_embs), dim=1)
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * labels.size(0)

                # Get classification performance
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            test_loss /= total
            test_accuracy = correct / total

            # Save if model had best accuracy so far
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                torch.save(model.state_dict(), "concat_model.pth")
            if VERBOSE: print(f"Epoch {epoch + 1}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Print the best performance over all 500 epochs
    print(best_test_acc)
    results['Concatenation'].append(best_test_acc)

import torch.nn.functional as F
from torch.nn import CosineSimilarity

# Simple contrastive learning model
class CLModel(nn.Module):
  def __init__(self):
    super(CLModel, self).__init__()

    # Projection network for peptide
    self.peptide_proj = torch.nn.Sequential(
      torch.nn.Linear(320, 128),
      torch.nn.ReLU(),
      torch.nn.Linear(128, 128),
    )

    # Projection network for protein
    self.protein_proj = torch.nn.Sequential(
      torch.nn.Linear(320, 128),
      torch.nn.ReLU(),
      torch.nn.Linear(128, 128),
    )

  def forward(self, pep, pro):
    # Project and normalize
    pep = self.peptide_proj(pep)
    pro = self.protein_proj(pro)
    pep = F.normalize(pep, p=2, dim=1)
    pro = F.normalize(pro, p=2, dim=1)
    return pep, pro

# Train the model 3 independent times. Report the highest performance on the test set each time.
for n in range(3):

    # Define model and other components
    model = CLModel()
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CosineEmbeddingLoss()

    best_test_acc = 0
    for epoch in tqdm(range(EPOCHS)):

        # Start training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # Load data for batch
        for batch in train_dataloader:
            peptide_embs, protein_embs, labels, _, _ = batch

            # Replace zeros with -1s for cosine loss
            labels[labels == 0] = -1

            # Forward pass
            peptide_embs, protein_embs, labels = peptide_embs.to(device), protein_embs.to(device), labels.to(device)
            pep_out, pro_out = model(peptide_embs, protein_embs)

            # Cosine similarity loss
            loss = criterion(pep_out, pro_out, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Get classification metrics via cosine similarity
            train_loss += loss.item() * labels.size(0)
            similarities = CosineSimilarity(dim=1, eps=1e-6)(pep_out, pro_out)
            preds = (similarities > 0).long() * 2 - 1  # Map to -1 or 1
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss /= total
        train_accuracy = correct / total

        # Evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_dataloader:
                peptide_embs, protein_embs, labels, _, _ = batch

                # Replace zeros with -1s for cosine loss
                labels[labels == 0] = -1

                # Forward pass, Cosine similarity loss
                peptide_embs, protein_embs, labels = peptide_embs.to(device), protein_embs.to(device), labels.to(device)
                pep_out, pro_out = model(peptide_embs, protein_embs)
                loss = criterion(pep_out, pro_out, labels)

                # Get classification metrics by cosine sim
                test_loss += loss.item() * labels.size(0)
                similarities = CosineSimilarity(dim=1, eps=1e-6)(pep_out, pro_out)
                preds = (similarities > 0).long() * 2 - 1 # Map to -1 or 1
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            test_loss /= total
            test_accuracy = correct / total

            # Save if model had best accuracy so far
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                torch.save(model.state_dict(), "contrastive_model.pth")
            if VERBOSE: print(f"Epoch {epoch + 1}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

    print(best_test_acc)
    results['Contrastive'].append(best_test_acc)

#@title Train attention model
EPOCHS=100
# Minimal cross attention network
class AttentionModel(nn.Module):
  def __init__(self, plm, PLM):
    super(AttentionModel, self).__init__()
    self.plm = plm
    self.PLM = PLM
    self.attn = nn.MultiheadAttention(320, 1, batch_first=True)
    self.prediction_head = torch.nn.Sequential(
        torch.nn.Linear(320, 2),
        # torch.nn.ReLU(),
        # torch.nn.Linear(128, 2),
    )
    self.post_attn_norm = nn.LayerNorm(320)
    # Freeze the anchor and augment models
    for param in self.plm.parameters(): param.requires_grad = False
    for param in self.PLM.parameters(): param.requires_grad = False

  def forward(self, pep_tokens, pro_tokens, attn_mask=None):

    # Obtain protein sequence embedding L x D. len(pro_tokens.input_ids) is len(protein_sequence)+2
    with torch.no_grad():
        results = self.PLM.forward(pro_tokens['input_ids'], pro_tokens['attention_mask'], output_hidden_states=True)
    t1 = results.hidden_states[-1]

    # Obtain peptide sequence embedding L x D. len(pep_tokens.input_ids) is len(peptide_sequence)+2
    with torch.no_grad():
        results = self.plm.forward(pep_tokens['input_ids'], pep_tokens['attention_mask'], output_hidden_states=True)
    t2 = results.hidden_states[-1]

    # Create attn mask
    attn_mask = torch.matmul(
        pro_tokens["attention_mask"].unsqueeze(2).float(),
        pep_tokens['attention_mask'].unsqueeze(1).float(),
    ).repeat(1, 1, 1)

    # Perform attention
    output, attn_weights = self.attn(
        query=t1, key=t2, value=t2, attn_mask=attn_mask, average_attn_weights=False
    )
    output = self.post_attn_norm(output) + t1

    # Take the mean. Exclude padding tokens and bos/eos
    mask_sum = pro_tokens['attention_mask'].sum(dim=1, keepdim=True).clamp(min=1e-6)  # Avoid division by zero
    output = (output * pro_tokens['attention_mask'].unsqueeze(2)).sum(dim=1) / mask_sum

    return self.prediction_head(output), output

for n in range(3):
    # Define model and other components
    model = AttentionModel(plm, PLM)
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_test_acc = 0
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_dataloader:
            # Forward pass
            _, _, labels, pep_tokens, pro_tokens = batch
            labels = labels.to(device)
            outputs, _ = model(pep_tokens, pro_tokens)
            loss = criterion(outputs, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Compute classification metrics
            train_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_loss /= total
        train_accuracy = correct / total

        # Evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_dataloader:

                # Forward pass
                _, _, labels, pep_tokens, pro_tokens = batch
                labels = labels.to(device)
                outputs, _ = model(pep_tokens, pro_tokens)
                loss = criterion(outputs, labels)

                # Compute classification metrics
                test_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            test_loss /= total
            test_accuracy = correct / total

        # Save if best
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            torch.save(model.state_dict(), "attention_model.pth")
        if VERBOSE: print(f"Epoch {epoch + 1}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print(best_test_acc)
    results['Attention'].append(best_test_acc)

#@title Train composition style model

# Minimal composition of language models
target_layers = [0, 3, 5]
class CompositionModel(nn.Module):
  def __init__(self, plm, PLM):
    super(CompositionModel, self).__init__()
    self.plm = plm
    self.PLM = PLM
    self.cross_atten_layers = nn.ModuleList([
        nn.MultiheadAttention(320, 20, batch_first=True) for i in range(len(target_layers))
    ])
    self.post_attn_norms = nn.ModuleList([
        nn.LayerNorm(320) for i in range(len(target_layers))
    ])
    self.prediction_head = torch.nn.Sequential(
        torch.nn.Linear(320, 2),
        # torch.nn.ReLU(),
        # torch.nn.Linear(128, 2),
    )
    self.esm_layers = 6
    # Freeze the anchor and augment models
    for param in self.plm.parameters(): param.requires_grad = False
    for param in self.PLM.parameters(): param.requires_grad = False

  def forward(self, pep_input, prot_input, attn_mask=None):
    # Create attn mask
    attn_mask = torch.matmul(
        prot_input["attention_mask"].unsqueeze(2).float(),
        pep_input['attention_mask'].unsqueeze(1).float(),
    ).repeat(20, 1, 1)

    # Create the esm attention masks correctly
    pep_attn_mask = self.plm.get_extended_attention_mask(pep_input['attention_mask'], pep_input["input_ids"].size())
    prot_attn_mask = self.PLM.get_extended_attention_mask(prot_input['attention_mask'], prot_input["input_ids"].size())

    # Embedding layer
    pro = self.PLM.esm.embeddings(prot_input["input_ids"], prot_input["attention_mask"]).to(device)
    with torch.no_grad():
        pep = self.plm.esm.embeddings(pep_input["input_ids"], pep_input["attention_mask"]).to(device)

    # Layerwise forward pass
    counter = 0
    for i in range(0, self.esm_layers):

        # Update embeddings
        pro = self.PLM.esm.encoder.layer[i](pro, prot_attn_mask)[0]#, prot_input["attention_mask"][:, None, None, :])[0]
        with torch.no_grad():
            pep = self.plm.esm.encoder.layer[i](pep, pep_attn_mask)[0]#, pep_input["attention_mask"][:, None, None, :])[0]

        if i in target_layers:
            # Perform cross attn and layer norm
            attn_out, _ = self.cross_atten_layers[counter](query=pro, key=pep, value=pep, attn_mask=attn_mask, average_attn_weights=False)
            attn_out = self.post_attn_norms[counter](attn_out)
            pro = pro + attn_out
            counter += 1

    pro = self.PLM.esm.encoder.emb_layer_norm_after(pro)

    # Take the mean. Exclude padding tokens and bos/eos
    mask_sum = prot_input['attention_mask'].sum(dim=1, keepdim=True).clamp(min=1e-6)  # Avoid division by zero
    pro = (pro * prot_input['attention_mask'].unsqueeze(2)).sum(dim=1) / mask_sum
    return self.prediction_head(pro), pro

for n in range(3):

    # Define model and other components
    model = CompositionModel(plm, PLM)
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_test_acc = 0
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_dataloader:
            # Forward pass
            _, _, labels, pep_tokens, pro_tokens = batch
            labels = labels.to(device)
            outputs, _ = model(pep_tokens, pro_tokens)
            loss = criterion(outputs, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Compute classification metrics
            train_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_loss /= total
        train_accuracy = correct / total

        # Evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_dataloader:
                # Forward pass
                _, _, labels, pep_tokens, pro_tokens = batch
                labels = labels.to(device)
                outputs, _ = model(pep_tokens, pro_tokens)
                loss = criterion(outputs, labels)
                # Compute classification metrics
                test_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        test_loss /= total
        test_accuracy = correct / total

        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            #torch.save(model.state_dict(), "comp_model.pth")
        if VERBOSE: print(f"Epoch {epoch + 1}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

    print(best_test_acc)
    results['Composition'].append(best_test_acc)

# Save results to CSV
df_results = pd.DataFrame({
    'Model': list(results.keys()),
    'Run 1': [results[k][0] if len(results[k]) > 0 else None for k in results.keys()],
    'Run 2': [results[k][1] if len(results[k]) > 1 else None for k in results.keys()],
    'Run 3': [results[k][2] if len(results[k]) > 2 else None for k in results.keys()],
})
df_results.to_csv('results_tannerHP.csv', index=False)
print("\nResults saved to results_tannerHP.csv")