#!/usr/bin/env python3
"""
Build LSTM-ready data from Tashlhiyt noun dataset.

This script:
1. Extracts character vocabulary from singular themes
2. Encodes sequences as integer arrays
3. Creates train/test splits matching existing ablation experiments
4. Saves preprocessed data for LSTM training

Usage:
    python scripts/build_lstm_data.py

Output:
    features/char_vocab.json - Character to integer mapping
    features/lstm_data_has_suffix.npz - Preprocessed data for has_suffix domain
    features/lstm_data_ablaut.npz - Preprocessed data for ablaut domain
    features/lstm_data_medial_a.npz - Preprocessed data for medial_a domain
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedKFold

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def extract_character_vocabulary(df, theme_column='analysisSingularTheme'):
    """
    Extract unique characters from singular themes and create vocabulary.

    Args:
        df: DataFrame with singular themes
        theme_column: Column name containing singular themes

    Returns:
        dict: Character to integer mapping (0 reserved for padding)
    """
    print("\n=== Extracting Character Vocabulary ===")

    # Get all non-null themes
    themes = df[theme_column].dropna().values

    # Count all characters
    all_chars = ''.join(themes)
    char_counts = Counter(all_chars)

    print(f"Total characters (with duplicates): {len(all_chars):,}")
    print(f"Unique characters: {len(char_counts)}")

    # Create vocabulary (0 reserved for padding)
    vocab = {'<PAD>': 0}
    for idx, char in enumerate(sorted(char_counts.keys()), start=1):
        vocab[char] = idx

    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Top 10 most frequent characters:")
    for char, count in char_counts.most_common(10):
        print(f"  '{char}': {count:,} occurrences (ID: {vocab[char]})")

    return vocab


def encode_sequences(themes, vocab, max_len=None, padding='post'):
    """
    Encode character sequences as integer arrays.

    Args:
        themes: List of theme strings
        vocab: Character to integer mapping
        max_len: Maximum sequence length (None = auto-detect)
        padding: 'post' or 'pre' padding

    Returns:
        np.ndarray: Encoded sequences (n_samples, max_len)
    """
    # Determine max length
    if max_len is None:
        max_len = max(len(theme) for theme in themes)

    # Initialize array
    encoded = np.zeros((len(themes), max_len), dtype=np.int32)

    # Encode each sequence
    for i, theme in enumerate(themes):
        # Convert characters to integers
        char_ids = [vocab.get(char, vocab['<PAD>']) for char in theme]

        # Truncate or pad
        seq_len = min(len(char_ids), max_len)
        if padding == 'post':
            encoded[i, :seq_len] = char_ids[:seq_len]
        else:  # pre
            encoded[i, -seq_len:] = char_ids[:seq_len]

    return encoded


def prepare_domain_data(df, domain_name, vocab, target_column, max_len):
    """
    Prepare data for a specific domain by loading pre-generated target files.

    Args:
        df: Full DataFrame
        domain_name: Domain name (e.g., 'has_suffix', 'ablaut', etc.)
        vocab: Character vocabulary
        target_column: Column name for target labels (not used - kept for compatibility)
        max_len: Maximum sequence length

    Returns:
        dict: Contains X (sequences), y (labels), themes (original strings),
              label_map (class names), and metadata
    """
    print(f"\n=== Preparing Data for {domain_name} ===")

    # Determine target file based on domain type
    if domain_name in ['has_suffix', 'has_mutation', '3way']:
        # Macro-level domains
        target_file = PROJECT_ROOT / 'features' / f'y_macro_{domain_name}.csv'
    else:
        # Micro-level domains
        target_file = PROJECT_ROOT / 'features' / f'y_micro_{domain_name}.csv'

    # Load target data
    if not target_file.exists():
        raise FileNotFoundError(f"Target file not found: {target_file}")

    y_df = pd.read_csv(target_file, index_col=0)  # recordID is the index

    # Merge with main dataset using the recordID indices
    # Only keep rows that exist in the target file
    common_indices = df.index.intersection(y_df.index)

    df_domain = df.loc[common_indices].copy()
    df_domain['target'] = y_df.loc[common_indices].iloc[:, 0].values  # First column is the target

    # Remove rows with missing singular themes
    df_domain = df_domain.dropna(subset=['analysisSingularTheme'])

    print(f"Dataset size: {len(df_domain)} nouns")

    # Handle string labels (convert to integers)
    if df_domain['target'].dtype == 'object':
        # Create label encoding
        unique_str_labels = sorted(df_domain['target'].unique())
        str_to_int = {label: i for i, label in enumerate(unique_str_labels)}
        label_map = {i: label for i, label in enumerate(unique_str_labels)}
        df_domain['target'] = df_domain['target'].map(str_to_int)
    else:
        # Numeric labels - create generic label map
        unique_labels = sorted(df_domain['target'].unique())
        if len(unique_labels) == 2:
            label_map = {0: 'Class 0', 1: 'Class 1'}
        else:
            label_map = {int(i): f'Class {int(i)}' for i in unique_labels}

    print(f"Class distribution:")
    for label_id, label_name in label_map.items():
        count = (df_domain['target'] == label_id).sum()
        pct = count / len(df_domain) * 100
        print(f"  {label_name}: {count} ({pct:.1f}%)")

    # Encode sequences
    themes = df_domain['analysisSingularTheme'].values
    X = encode_sequences(themes, vocab, max_len=max_len)
    y = df_domain['target'].values

    # Get original indices for fold consistency
    original_indices = df_domain.index.values

    print(f"Encoded sequences shape: {X.shape}")
    print(f"Max sequence length: {max_len}")

    return {
        'X': X,
        'y': y,
        'themes': themes,
        'label_map': label_map,
        'original_indices': original_indices,
        'n_classes': len(label_map),
        'domain_name': domain_name
    }


def create_folds(y, n_splits=10, random_state=42):
    """
    Create stratified k-fold splits.

    Args:
        y: Target labels
        n_splits: Number of folds
        random_state: Random seed for reproducibility

    Returns:
        list: List of (train_idx, val_idx) tuples
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(skf.split(np.zeros(len(y)), y))

    print(f"\n=== Created {n_splits}-Fold Stratified Splits ===")
    print(f"Random seed: {random_state}")

    # Verify stratification
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_dist = np.bincount(y[train_idx]) / len(train_idx)
        val_dist = np.bincount(y[val_idx]) / len(val_idx)

        # Handle extreme imbalance where positive class might be missing in some folds
        train_pos_pct = train_dist[1] if len(train_dist) > 1 else 0.0
        val_pos_pct = val_dist[1] if len(val_dist) > 1 else 0.0

        print(f"Fold {fold_idx}: Train {len(train_idx)} ({train_pos_pct:.1%} positive), "
              f"Val {len(val_idx)} ({val_pos_pct:.1%} positive)")

    return folds


def main():
    """Main preprocessing pipeline."""

    print("=" * 80)
    print("LSTM Data Preprocessing Pipeline")
    print("=" * 80)

    # Paths
    data_path = PROJECT_ROOT / 'data' / 'tash_nouns.csv'
    vocab_path = PROJECT_ROOT / 'features' / 'char_vocab.json'
    features_dir = PROJECT_ROOT / 'features'
    features_dir.mkdir(exist_ok=True)

    # Load dataset
    print(f"\nLoading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Total records: {len(df)}")

    # Extract vocabulary
    vocab = extract_character_vocabulary(df)

    # Save vocabulary
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"\nSaved vocabulary to: {vocab_path}")

    # Determine max sequence length across all themes
    all_themes = df['analysisSingularTheme'].dropna().values
    max_len = max(len(theme) for theme in all_themes)
    print(f"\nMaximum sequence length: {max_len}")

    # Domains to process
    # All 10 domains for LSTM training
    domains = [
        # Macro-level (3 domains)
        ('has_suffix', 'analysisPluralPattern'),
        ('has_mutation', 'analysisPluralPattern'),
        ('3way', 'analysisPluralPattern'),
        # Micro-level (7 domains)
        ('ablaut', 'analysisInternalChanges'),
        ('medial_a', 'analysisInternalChanges'),
        ('final_a', 'analysisInternalChanges'),
        ('final_vw', 'analysisInternalChanges'),
        ('insert_c', 'analysisInternalChanges'),
        ('templatic', 'analysisInternalChanges'),
        ('8way', 'analysisInternalChanges')
    ]

    # Process each domain
    for domain_name, target_col in domains:
        # Prepare data
        data = prepare_domain_data(df, domain_name, vocab, target_col, max_len)

        # Create folds
        folds = create_folds(data['y'], n_splits=10, random_state=42)

        # Save preprocessed data
        output_path = features_dir / f"lstm_data_{domain_name}.npz"
        np.savez_compressed(
            output_path,
            X=data['X'],
            y=data['y'],
            themes=data['themes'],
            original_indices=data['original_indices'],
            label_map=json.dumps(data['label_map']),
            n_classes=data['n_classes'],
            vocab_size=len(vocab),
            max_len=max_len,
            domain_name=domain_name,
            folds=np.array(folds, dtype=object)  # Save fold indices
        )
        print(f"Saved preprocessed data to: {output_path}")

    print("\n" + "=" * 80)
    print("Preprocessing Complete!")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  - {vocab_path}")
    print(f"  - {features_dir / 'lstm_data_has_suffix.npz'}")
    print(f"  - {features_dir / 'lstm_data_ablaut.npz'}")
    print(f"  - {features_dir / 'lstm_data_medial_a.npz'}")
    print(f"\nNext step: Run experiments/train_lstm.py")


if __name__ == '__main__':
    main()
