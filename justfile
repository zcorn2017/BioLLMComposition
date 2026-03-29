# precompute → split → train (run from repo root)
# just precompute [config=...] | just split [config=...] [data_pt=...] | just train [config=...] [split_pt=...]

py := 'PYTHONPATH=. python'
precompute_config := 'configs/precompute/default.yaml'
split_config      := 'configs/splitting/default.yaml'
train_config      := 'configs/training/composition_contactmap_focal_loss.yaml'
embeddings_dir    := '/home/zcorn/Projects/proteinDNA_data/processed/embeddings'
split_dir         := '/home/zcorn/Projects/proteinDNA_data/processed/split_dataset'

default:
    {{py}} scripts/00_precompute_tokens_and_labels.py --config {{precompute_config}}
    {{py}} scripts/01_split_datasets.py --config {{split_config}} \
        --data_pt $(shell ls -t "{{embeddings_dir}}"/tokens_*.pt 2>/dev/null | head -1) \
        --out_dir "{{split_dir}}"
    {{py}} scripts/02_run_training.py --config {{train_config}} \
        --split_pt $(shell ls -t "{{split_dir}}"/split_*.pt 2>/dev/null | head -1)

precompute config=precompute_config:
    {{py}} scripts/00_precompute_tokens_and_labels.py --config {{config}}

split config=split_config data_pt='':
    {{py}} scripts/01_split_datasets.py --config {{config}} \
        $(shell [ -n '{{data_pt}}' ] && echo '--data_pt' '{{data_pt}}') \
        --out_dir "{{split_dir}}"

train config=train_config split_pt='':
    {{py}} scripts/02_run_training.py --config {{config}} \
        $(shell [ -n '{{split_pt}}' ] && echo '--split_pt' '{{split_pt}}')

run:
    python scripts/frameworks/composition_contactmap_v2.py \
    --config configs/training/composition_contactmap_v2.yaml \
    --contact_head refined --dynamic_batching
    python scripts/frameworks/composition_contactmap_v2.py \
    --config configs/training/composition_contactmap_v2.yaml