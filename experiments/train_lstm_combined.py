#!/usr/bin/env python3
"""
Train Combined Bi-LSTM model (LSTM + Morph+Phon features).

This script trains the hybrid architecture that concatenates LSTM representations
with hand-crafted morphological and phonological features.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

# Add experiments directory to path
sys.path.insert(0, str(Path(__file__).parent))

from lstm_model import build_combined_model
from lstm_utils import (
    load_lstm_data,
    compute_metrics,
    save_fold_results,
    create_summary_csv,
)


def load_morph_phon_features(domain_name):
    """
    Load Morph+Phon feature matrix for the domain.

    Args:
        domain_name: Domain name (e.g., 'has_suffix', 'ablaut', 'medial_a')

    Returns:
        tuple: (X_features, original_indices) where X_features is (n_samples, n_features)
    """
    # Load from ablation directory
    features_dir = Path('features') / f'ablation_{domain_name}'
    feature_file = features_dir / 'X_morph_phon.csv'

    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")

    # Load features
    df = pd.read_csv(feature_file)

    # Extract record IDs and features
    record_ids = df['recordID'].values
    feature_cols = [col for col in df.columns if col != 'recordID']
    X_features = df[feature_cols].values.astype(np.float32)

    print(f"  Loaded Morph+Phon features: {X_features.shape}")
    print(f"  Feature columns: {len(feature_cols)}")

    return X_features, record_ids




def train_single_fold(
    fold_idx,
    X_char_train, X_char_val,
    X_feat_train, X_feat_val,
    y_train, y_val,
    themes_val,
    vocab_size,
    max_len,
    num_features,
    num_classes,
    verbose=1
):
    """
    Train a single fold of the combined model.

    Args:
        fold_idx: Fold index
        X_char_train, X_char_val: Character sequences (train/val)
        X_feat_train, X_feat_val: Feature matrices (train/val)
        y_train, y_val: Labels (train/val)
        themes_val: Original theme strings for validation
        vocab_size: Size of character vocabulary
        max_len: Maximum sequence length
        num_features: Number of hand-crafted features
        num_classes: Number of output classes
        verbose: Verbosity level

    Returns:
        tuple: (y_val, y_pred, y_pred_proba, themes_val, history, metrics)
    """
    print(f"  Building Combined model (LSTM + Morph+Phon)...")

    # Build model
    model = build_combined_model(
        vocab_size=vocab_size,
        max_len=max_len,
        num_features=num_features,
        num_classes=num_classes,
        embedding_dim=32,
        lstm_units=64,
        dropout=0.3
    )

    if fold_idx == 0 and verbose:
        print("\n  Model Architecture:")
        model.summary()

    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=verbose
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=verbose,
        min_lr=1e-6
    )

    # Train
    print("  Training...")
    history = model.fit(
        [X_char_train, X_feat_train], y_train,
        validation_data=([X_char_val, X_feat_val], y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=verbose
    )

    # Predict on validation
    print("  Predicting on validation set...")
    y_pred_proba = model.predict([X_char_val, X_feat_val], verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Compute metrics
    metrics = compute_metrics(y_val, y_pred, y_pred_proba)

    print(f"  Results:")
    print(f"    Accuracy:  {metrics['accuracy']:.3f}")
    print(f"    Macro-F1:  {metrics['macro_f1']:.3f}")
    print(f"    AUC-ROC:   {metrics['auc_roc']:.3f}")
    print(f"    Epochs trained: {len(history.history['loss'])}")

    return y_val, y_pred, y_pred_proba, themes_val, history, metrics


def train_domain(
    domain_name,
    data_path=None,
    results_dir=None,
    verbose=1
):
    """
    Train combined model for a single domain with 10-fold cross-validation.

    Uses ablation feature files as the source of truth for record IDs and labels.

    Args:
        domain_name: Domain name (e.g., 'has_suffix', 'ablaut', 'medial_a')
        data_path: Path to LSTM data .npz file (default: auto-detect)
        results_dir: Directory to save results (default: auto-detect)
        verbose: Verbosity level (0=silent, 1=progress, 2=debug)

    Returns:
        dict: Summary statistics
    """
    print("=" * 60)
    print(f"Training Combined Bi-LSTM for domain: {domain_name}")
    print("=" * 60)
    print()

    # Load Morph+Phon features (source of truth for record IDs)
    print("Loading Morph+Phon features...")
    X_features, feature_record_ids = load_morph_phon_features(domain_name)
    num_features = X_features.shape[1]
    num_samples = len(X_features)

    print(f"  Features: {X_features.shape}")
    print(f"  Record IDs: {len(feature_record_ids)}")
    print()

    # Load main dataset to get character sequences
    print("Loading main dataset for character sequences...")
    df = pd.read_csv('data/tash_nouns.csv')

    # Create character encoding from scratch
    print("Creating character vocabulary...")
    all_themes = df['analysisSingularTheme'].dropna().values
    vocab = sorted(set(''.join(all_themes)))
    char_to_idx = {c: idx + 1 for idx, c in enumerate(vocab)}  # 0 reserved for padding
    vocab_size = len(vocab) + 1

    # Determine max length
    max_len = max(len(theme) for theme in all_themes)

    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Max sequence length: {max_len}")
    print()

    # Encode character sequences for our samples
    print("Encoding character sequences...")
    X_char_list = []
    y_list = []
    themes_list = []

    # Load target variable
    y_file = Path('features') / f'ablation_{domain_name}' / f'y_{domain_name}.csv'
    y_df = pd.read_csv(y_file)

    # Create mapping from recordID to label (column name varies by domain)
    label_col = [col for col in y_df.columns if col != 'recordID'][0]
    record_to_label = dict(zip(y_df['recordID'], y_df[label_col]))

    for rec_id in feature_record_ids:
        # Get row from main dataset
        row = df[df['recordID'] == rec_id]
        if len(row) == 0:
            raise ValueError(f"Record {rec_id} not found in main dataset")

        theme = row['analysisSingularTheme'].iloc[0]
        label = record_to_label[rec_id]

        # Encode theme
        encoded = [char_to_idx.get(c, 0) for c in theme]
        # Pad to max_len
        padded = encoded + [0] * (max_len - len(encoded))

        X_char_list.append(padded)
        y_list.append(label)
        themes_list.append(theme)

    X_char = np.array(X_char_list, dtype=np.int32)
    y = np.array(y_list, dtype=np.int32)
    themes = np.array(themes_list)

    num_classes = len(np.unique(y))
    label_map = {str(i): f'Class_{i}' for i in range(num_classes)}

    print(f"  Encoded sequences: {X_char.shape}")
    print(f"  Labels: {y.shape}")
    print(f"  Num classes: {num_classes}")
    print()

    # Create 10-fold split (same as ablation)
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    folds_list = []
    for train_idx, val_idx in skf.split(X_char, y):
        folds_list.append((train_idx, val_idx))
    folds = np.array(folds_list, dtype=object)

    print(f"  Created 10-fold CV splits")
    print()

    # Setup results directory
    if results_dir is None:
        results_dir = Path('experiments/results') / f'lstm_combined_{domain_name}'
    else:
        results_dir = Path(results_dir)

    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    print()

    # 10-fold cross-validation
    print("Running 10-fold cross-validation...")
    print()

    all_results = []

    for fold_idx in range(10):
        print("=" * 60)
        print(f"Fold {fold_idx}")
        print("=" * 60)

        # Get train/val indices for this fold
        train_idx, val_idx = folds[fold_idx]

        # Split data
        X_char_train, X_char_val = X_char[train_idx], X_char[val_idx]
        X_feat_train, X_feat_val = X_features[train_idx], X_features[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        themes_val = themes[val_idx]

        print(f"  Train size: {len(X_char_train)}, Val size: {len(X_char_val)}")
        print(f"  Train class distribution: {np.bincount(y_train)}")
        print(f"  Val class distribution: {np.bincount(y_val)}")

        # Train fold
        y_val_true, y_pred, y_pred_proba, themes_val, history, metrics = train_single_fold(
            fold_idx=fold_idx,
            X_char_train=X_char_train,
            X_char_val=X_char_val,
            X_feat_train=X_feat_train,
            X_feat_val=X_feat_val,
            y_train=y_train,
            y_val=y_val,
            themes_val=themes_val,
            vocab_size=vocab_size,
            max_len=max_len,
            num_features=num_features,
            num_classes=num_classes,
            verbose=verbose
        )

        # Save fold results
        save_fold_results(
            results_dir, fold_idx,
            y_val_true, y_pred, y_pred_proba,
            themes_val, history, metrics
        )

        all_results.append(metrics)
        print()

    # Create summary
    print("=" * 60)
    print("Creating summary statistics...")
    print()

    summary_df = create_summary_csv(results_dir, domain_name, 'lstm_combined')

    return summary_df


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(
        description='Train Combined Bi-LSTM model (LSTM + Morph+Phon features)'
    )
    parser.add_argument(
        '--domain',
        type=str,
        required=True,
        choices=[
            'has_suffix', 'has_mutation', '3way',
            'medial_a', 'final_a', 'final_vw', 'ablaut', 'insert_c', 'templatic', '8way'
        ],
        help='Domain to train on'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to LSTM data .npz file (default: auto-detect)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Directory to save results (default: auto-detect)'
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help='Verbosity level (0=silent, 1=progress, 2=debug)'
    )

    args = parser.parse_args()

    # Print TensorFlow configuration
    import tensorflow as tf
    print("=" * 80)
    print("TensorFlow Configuration")
    print("=" * 80)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    print(f"Metal (M4) available: {tf.config.list_physical_devices('Metal')}")
    print("=" * 80)
    print()

    # Record start time
    start_time = time.time()

    # Train domain
    summary = train_domain(
        domain_name=args.domain,
        data_path=args.data_path,
        results_dir=args.results_dir,
        verbose=args.verbose
    )

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    elapsed_mins = elapsed_time / 60

    # Print completion message
    print()
    print("=" * 80)
    print(f"✓ Training complete for {args.domain}")
    print(f"✓ Results saved to: experiments/results/lstm_combined_{args.domain}")
    print("=" * 80)
    print()
    print(f"Domain '{args.domain}' completed in {elapsed_mins:.1f} minutes")
    print()
    print("=" * 80)
    print("ALL TRAINING COMPLETE")
    print("=" * 80)
    print()
    print(f"Total time: {elapsed_mins:.1f} minutes")
    print()
    print(f"Domains trained: {args.domain}")
    print()
    print("=== Performance Summary ===")
    print()
    print(f"{args.domain}:")
    print(f"  Accuracy:  {summary['accuracy_mean'].iloc[0]:.3f} ± {summary['accuracy_std'].iloc[0]:.3f}")
    print(f"  Macro-F1:  {summary['macro_f1_mean'].iloc[0]:.3f} ± {summary['macro_f1_std'].iloc[0]:.3f}")
    print(f"  AUC-ROC:   {summary['auc_roc_mean'].iloc[0]:.3f} ± {summary['auc_roc_std'].iloc[0]:.3f}")
    print()
    print("=" * 80)
    print("Next steps:")
    print("  1. Review results in experiments/results/lstm_combined_*/")
    print("  2. Compare with LSTM baseline and feature-based models")
    print("  3. Perform residual analysis (Day 2)")
    print("=" * 80)
    print()


if __name__ == '__main__':
    main()
