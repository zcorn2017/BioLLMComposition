"""
Cluster-based Data Splitting for Protein-DNA Dataset

This script performs train/val/test splitting on the protein-DNA dataset
using cluster_id to prevent data leakage. Proteins were clustered using:
    mmseqs easy-cluster proteins_for_clustering.fasta cluster_result tmp \
        --min-seq-id 0.3 -c 0.8 --cov-mode 1

This ensures that proteins from the same homology cluster (>30% sequence identity)
don't appear in both train and test sets, preventing data leakage from
evolutionarily related proteins.

Usage:
    python scripts/split_protein_dna_data.py
    
    # Or with custom split ratios:
    python scripts/split_protein_dna_data.py --test-size 0.15 --val-size 0.15
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit


def parse_args():
    parser = argparse.ArgumentParser(
        description='Split protein-DNA dataset by cluster to prevent data leakage'
    )
    parser.add_argument(
        '--data-path', 
        type=str,
        default='/home/zcorn/Projects/proteinDNA_data/working/dnaprodb2/dna_protein_chain_wise_cleaned.parquet',
        help='Path to the input parquet file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/zcorn/Projects/proteinDNA_data/working/dnaprodb2/splits',
        help='Directory to save the split datasets'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for test set (default: 0.2)'
    )
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.0,
        help='Proportion of training data for validation set (default: 0.0, no validation split)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['parquet', 'csv', 'both'],
        default='parquet',
        help='Output format (default: parquet)'
    )
    return parser.parse_args()


def verify_no_cluster_overlap(train_df, val_df, test_df):
    """Verify that there is no cluster overlap between splits."""
    train_clusters = set(train_df['cluster_id'].unique())
    test_clusters = set(test_df['cluster_id'].unique())
    
    train_test_overlap = train_clusters & test_clusters
    if train_test_overlap:
        raise ValueError(f"Cluster overlap between train and test: {len(train_test_overlap)} clusters")
    
    if val_df is not None and len(val_df) > 0:
        val_clusters = set(val_df['cluster_id'].unique())
        train_val_overlap = train_clusters & val_clusters
        val_test_overlap = val_clusters & test_clusters
        
        if train_val_overlap:
            raise ValueError(f"Cluster overlap between train and val: {len(train_val_overlap)} clusters")
        if val_test_overlap:
            raise ValueError(f"Cluster overlap between val and test: {len(val_test_overlap)} clusters")
    
    print("✓ No cluster overlap detected between splits")


def print_split_stats(train_df, val_df, test_df):
    """Print statistics about the data splits."""
    print("\n" + "="*60)
    print("Dataset Split Statistics")
    print("="*60)
    
    total = len(train_df) + (len(val_df) if val_df is not None else 0) + len(test_df)
    
    print(f"\nTotal samples: {total}")
    print(f"  Train: {len(train_df)} ({100*len(train_df)/total:.1f}%)")
    if val_df is not None and len(val_df) > 0:
        print(f"  Val:   {len(val_df)} ({100*len(val_df)/total:.1f}%)")
    print(f"  Test:  {len(test_df)} ({100*len(test_df)/total:.1f}%)")
    
    print(f"\nCluster distribution:")
    print(f"  Train clusters: {train_df['cluster_id'].nunique()}")
    if val_df is not None and len(val_df) > 0:
        print(f"  Val clusters:   {val_df['cluster_id'].nunique()}")
    print(f"  Test clusters:  {test_df['cluster_id'].nunique()}")
    
    print(f"\nLabel distribution:")
    print("  Train:")
    for label, count in train_df['label'].value_counts().items():
        print(f"    Label {label}: {count} ({100*count/len(train_df):.1f}%)")
    
    if val_df is not None and len(val_df) > 0:
        print("  Val:")
        for label, count in val_df['label'].value_counts().items():
            print(f"    Label {label}: {count} ({100*count/len(val_df):.1f}%)")
    
    print("  Test:")
    for label, count in test_df['label'].value_counts().items():
        print(f"    Label {label}: {count} ({100*count/len(test_df):.1f}%)")


def main():
    args = parse_args()
    
    print(f"Loading data from: {args.data_path}")
    df = pd.read_parquet(args.data_path)
    print(f"Loaded {len(df)} samples with {df['cluster_id'].nunique()} clusters")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(args.seed)
    
    # First split: train+val vs test
    print(f"\nSplitting data with test_size={args.test_size}, seed={args.seed}")
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_val_idx, test_idx = next(gss.split(df, groups=df['cluster_id']))
    
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    
    # Second split: train vs val (if requested)
    val_df = None
    if args.val_size > 0:
        print(f"Further splitting train for validation with val_size={args.val_size}")
        gss_val = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
        train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df['cluster_id']))
        
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
    else:
        train_df = train_val_df
    
    # Verify no overlap
    verify_no_cluster_overlap(train_df, val_df, test_df)
    
    # Print statistics
    print_split_stats(train_df, val_df, test_df)
    
    # Save splits
    print(f"\nSaving splits to: {output_dir}")
    
    if args.format in ['parquet', 'both']:
        train_df.to_parquet(output_dir / 'train.parquet', index=False)
        test_df.to_parquet(output_dir / 'test.parquet', index=False)
        if val_df is not None and len(val_df) > 0:
            val_df.to_parquet(output_dir / 'val.parquet', index=False)
        print("  ✓ Saved parquet files")
    
    if args.format in ['csv', 'both']:
        train_df.to_csv(output_dir / 'train.csv', index=False)
        test_df.to_csv(output_dir / 'test.csv', index=False)
        if val_df is not None and len(val_df) > 0:
            val_df.to_csv(output_dir / 'val.csv', index=False)
        print("  ✓ Saved CSV files")
    
    # Save split info
    split_info = {
        'data_path': str(args.data_path),
        'test_size': args.test_size,
        'val_size': args.val_size,
        'seed': args.seed,
        'total_samples': len(df),
        'train_samples': len(train_df),
        'val_samples': len(val_df) if val_df is not None else 0,
        'test_samples': len(test_df),
        'total_clusters': df['cluster_id'].nunique(),
        'train_clusters': train_df['cluster_id'].nunique(),
        'val_clusters': val_df['cluster_id'].nunique() if val_df is not None else 0,
        'test_clusters': test_df['cluster_id'].nunique(),
    }
    
    import json
    with open(output_dir / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    print("  ✓ Saved split_info.json")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    main()
