## Integrating Large Language Models to Learn Biophysical Interactions [[Preprint](https://arxiv.org/abs/2503.21017)]

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjoecclark/BioLLMComposition/blob/main/BioLLMComposition.ipynb)

![TOC](toc.jpg) 

<div align="justify">
Large language models (LLMs) trained on biochemical sequences learn feature vectors that guide drug discovery through virtual screening. However, LLMs do not capture the molecular interactions important for binding affinity and specificity prediction. We compare a variety of methods to combine representations from distinct biological modalities to effectively represent molecular complexes. We demonstrate that learning to merge the representations from the internal layers of domain specific biological language models outperforms standard molecular interactions representations despite having significantly fewer features. 
</div>

### Quick Start
Our [Google Colab Notebook](https://colab.research.google.com/github/jjoecclark/BioLLMComposition/blob/main/BioLLMComposition.ipynb) compares and visualizes embeddings from four multimodal representation strategies.

### Citation:
```
@misc{BioLLMComposition.
      title={Two for the Price of One: Integrating Large Language Models to Learn Biophysical Interactions}, 
      author={Joseph D. Clark and Tanner J. Dean and Diwakar Shukla},
      year={2025},
      eprint={2503.21017},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2503.21017}, 
}
```


20260211
Code to train
```python
/home/zcorn/anaconda3/envs/py314-llm/bin/python /home/zcorn/Projects/BioLLMComposition/train_all_models_tannerHP_ProteinDNA.py 2>&1 | tee tanner_HP_Pro_DNA_20260211.log
```
Change LR from 5e-4 to 5e-3.

20260225
BioLLMComposition/
├── pyproject.toml                          # updated: find biollmcomposition*
├── biollmcomposition/
│   ├── __init__.py                         # new (empty)
│   ├── config.py                           # untouched
│   └── utils/
│       ├── __init__.py                     # new (empty)
│       └── contact_map.py                  # new — shared utilities
├── scripts/
│   ├── 00_precompute_tokens_and_labels.py  # new — preprocessing
│   └── frameworks/
│       ├── attention_contactmap.py         # new — attention training
│       └── composition_contactmap.py       # new — composition training



# 1. Precompute tokens + labels
python scripts/00_precompute_tokens_and_labels.py \
    --data_path /path/to/residue_wise_fullseq.pkl

# 2. Train attention model
python scripts/frameworks/attention_contactmap.py \
    --data_pt /path/to/contactmap_tokens_labels_*.pt --epochs 100

# 3. Train composition model
python scripts/frameworks/composition_contactmap.py \
    --data_pt /path/to/contactmap_tokens_labels_*.pt --epochs 100


diff: comp_cm_ntv3-100M_esmc-600M_groupshuffle_lr5e-05_bs16_hd64_nh16_tl253034_focal_a0.95_g2.0_20260320_111245 vs comp_cm_ntv3-100M_esmc-600M_groupshuffle_lr5e-05_bs16_hd64_nh16_tl253034_focal_a0.95_g2.0_20260320_111245 ntv3 changed to first layer.

comp_cm_ntv3-100M_esmc-600M_groupshuffle_lr5e-05_bs16_hd64_nh16_tl0510_focal_a0.95_g2.0_20260319_122013 vs 
runs/comp_cm_ntv3-100M_esmc-600M_groupshuffle_lr5e-05_bs16_hd64_nh16_tl253034_focal_a0.95_g2.0_20260320_004337 changed injected layers tl0510 to tl253034



Local (RTX 5090)
# Single experiment
make run-local JOB=composition_focal
make run-local JOB=composition_focal EXTRA="--focal_alpha 0.9 --focal_gamma 3.0"
make run-local JOB=attention_focal EXTRA="--lr 1e-4 --epochs 100"
# All available JOBs: attention_bce, attention_focal, composition_bce, composition_focal
Runs directly via python -u. ClearML env vars are injected inline. Results appear in the web UI immediately.

Biocluster (default TARGET)
# First-time setup (once)
make setup                          # pushes code + data + creates conda env
# Before each batch of experiments
make push                           # sync latest code
# Submit
make submit JOB=composition_focal
make submit JOB=composition_focal EXTRA="--focal_alpha 0.9 --focal_gamma 3.0"
make submit JOB=attention_focal
# Monitor
make status                          # SLURM queue
make logs                            # list recent log files
make logs JOB_ID=12345               # tail specific job's output
# Retrieve checkpoints (optional -- also in ClearML)
make pull
# Cancel
make cancel JOB_ID=12345
zcorn-compute
Same commands, add TARGET=compute:

make push TARGET=compute
make submit JOB=composition_focal TARGET=compute
make submit JOB=composition_focal TARGET=compute EXTRA="--focal_alpha 0.9"
make status TARGET=compute
make logs TARGET=compute
make pull TARGET=compute
Sweep (all targets)
# Focal loss grid search via the existing sweep script
bash slurm/sweep-focal-loss.sh                    # biocluster (default)
TARGET=compute bash slurm/sweep-focal-loss.sh      # zcorn-compute
bash slurm/sweep-focal-loss.sh --dry-run           # preview without submitting
ClearML server management
make server-status     # check containers
make server-logs       # tail API logs
make server-stop       # stop server
make server-start      # restart server
Web UI
Browse all experiments (from any machine) at: http://yumingz5-linux.scs.illinois.edu:9080

