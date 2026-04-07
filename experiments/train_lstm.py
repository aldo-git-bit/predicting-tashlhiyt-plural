#!/usr/bin/env python3
"""
Train Bi-LSTM models for Tashlhiyt plural prediction.

Trains baseline LSTM models using 10-fold cross-validation for:
- has_suffix (macro-level)
- ablaut (micro-level, high predictability)
- medial_a (micro-level, low predictability)

Usage:
    python experiments/train_lstm.py --domain has_suffix
    python experiments/train_lstm.py --domain ablaut
    python experiments/train_lstm.py --domain medial_a
    python experiments/train_lstm.py --all  # Train all 3 domains

Output:
    experiments/results/lstm_baseline_{domain}/
        - fold_0.json through fold_9.json (metrics)
        - predictions_fold_0.npz through predictions_fold_9.npz (predictions)
        - history_fold_0.json through history_fold_9.json (training curves)
        - lstm_summary.csv (aggregated results)
"""

import argparse
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime

# Add experiments directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lstm_model import build_bilstm
from lstm_utils import (
    load_lstm_data,
    compute_metrics,
    save_fold_results,
    create_summary_csv,
    apply_smote_if_needed
)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def train_single_fold(
    fold_idx,
    train_idx,
    val_idx,
    X,
    y,
    themes,
    original_indices,
    vocab_size,
    max_len,
    n_classes,
    use_smote=False,
    epochs=100,
    batch_size=32,
    patience=10,
    verbose=1
):
    """
    Train model on a single fold.

    Args:
        fold_idx: Fold index
        train_idx: Training indices
        val_idx: Validation indices
        X: Full feature matrix
        y: Full labels
        themes: Full theme strings
        original_indices: Original record IDs
        vocab_size: Character vocabulary size
        max_len: Maximum sequence length
        n_classes: Number of classes
        use_smote: Whether to apply SMOTE to training data
        epochs: Maximum epochs
        batch_size: Batch size
        patience: Early stopping patience
        verbose: Verbosity level

    Returns:
        tuple: (y_true_val, y_pred_val, y_pred_proba_val, themes_val, record_ids_val, history, metrics)
    """
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx}")
    print(f"{'='*60}")

    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    themes_val = themes[val_idx]
    record_ids_val = original_indices[val_idx]

    print(f"  Train size: {len(X_train)}, Val size: {len(X_val)}")
    print(f"  Train class distribution: {np.bincount(y_train)}")
    print(f"  Val class distribution: {np.bincount(y_val)}")

    # Apply SMOTE if requested
    if use_smote:
        X_train, y_train = apply_smote_if_needed(X_train, y_train, min_samples_threshold=50)

    # Build model
    print(f"  Building Bi-LSTM model...")
    model = build_bilstm(
        vocab_size=vocab_size,
        max_len=max_len,
        num_classes=n_classes,
        embedding_dim=32,
        lstm_units=64,
        dropout=0.3
    )

    if fold_idx == 0 and verbose > 0:
        print("\n  Model Architecture:")
        model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train
    print(f"  Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )

    # Predict
    print(f"  Predicting on validation set...")
    y_pred_proba_val = model.predict(X_val, verbose=0)
    y_pred_val = np.argmax(y_pred_proba_val, axis=1)

    # Compute metrics
    metrics = compute_metrics(y_val, y_pred_val, y_pred_proba_val)

    print(f"  Results:")
    print(f"    Accuracy:  {metrics['accuracy']:.3f}")
    print(f"    Macro-F1:  {metrics['macro_f1']:.3f}")
    print(f"    AUC-ROC:   {metrics['auc_roc']:.3f}")
    print(f"    Epochs trained: {len(history.history['loss'])}")

    return y_val, y_pred_val, y_pred_proba_val, themes_val, record_ids_val, history, metrics


def train_domain(domain_name, use_smote=False, epochs=100, batch_size=32, patience=10, verbose=1):
    """
    Train LSTM model for a domain using 10-fold CV.

    Args:
        domain_name: 'has_suffix', 'ablaut', or 'medial_a'
        use_smote: Whether to apply SMOTE
        epochs: Maximum epochs
        batch_size: Batch size
        patience: Early stopping patience
        verbose: Verbosity level
    """
    print("\n" + "="*80)
    print(f"Training Bi-LSTM for domain: {domain_name}")
    print("="*80)

    # Load data
    data_path = PROJECT_ROOT / 'features' / f'lstm_data_{domain_name}.npz'
    print(f"\nLoading data from: {data_path}")
    data = load_lstm_data(data_path)

    print(f"Dataset: {len(data['X'])} samples")
    print(f"Vocabulary size: {data['vocab_size']}")
    print(f"Max sequence length: {data['max_len']}")
    print(f"Number of classes: {data['n_classes']}")
    print(f"Label mapping: {data['label_map']}")

    # Results directory
    results_dir = PROJECT_ROOT / 'experiments' / 'results' / f'lstm_baseline_{domain_name}'
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {results_dir}")

    # Get folds
    folds = data['folds']
    print(f"\nRunning 10-fold cross-validation...")

    # Train each fold
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        y_val, y_pred, y_pred_proba, themes_val, record_ids_val, history, metrics = train_single_fold(
            fold_idx=fold_idx,
            train_idx=train_idx,
            val_idx=val_idx,
            X=data['X'],
            y=data['y'],
            themes=data['themes'],
            original_indices=data['original_indices'],
            vocab_size=data['vocab_size'],
            max_len=data['max_len'],
            n_classes=data['n_classes'],
            use_smote=use_smote,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            verbose=verbose
        )

        # Save fold results
        save_fold_results(
            results_dir=results_dir,
            fold_idx=fold_idx,
            y_true=y_val,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            themes=themes_val,
            history=history,
            metrics=metrics,
            record_ids=record_ids_val
        )

    # Create summary CSV
    print("\n" + "="*80)
    print("Creating summary statistics...")
    summary_df = create_summary_csv(results_dir, domain_name, 'lstm')

    print("\n" + "="*80)
    print(f"✓ Training complete for {domain_name}")
    print(f"✓ Results saved to: {results_dir}")
    print("="*80)

    return summary_df


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train Bi-LSTM models for Tashlhiyt plural prediction')
    parser.add_argument('--domain', type=str,
                        choices=['has_suffix', 'has_mutation', '3way', 'ablaut', 'medial_a',
                                'final_a', 'final_vw', 'insert_c', 'templatic', '8way'],
                        help='Domain to train')
    parser.add_argument('--all', action='store_true',
                        help='Train all 10 domains')
    parser.add_argument('--use-smote', action='store_true',
                        help='Apply SMOTE to training data')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2],
                        help='Verbosity level (default: 1)')

    args = parser.parse_args()

    # Check TensorFlow GPU availability
    print("="*80)
    print("TensorFlow Configuration")
    print("="*80)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    print(f"Metal (M4) available: {tf.config.list_physical_devices('GPU')}")
    print("="*80)

    # Determine domains to train
    if args.all:
        # All 10 domains
        domains = ['has_suffix', 'has_mutation', '3way', 'ablaut', 'medial_a',
                  'final_a', 'final_vw', 'insert_c', 'templatic', '8way']
    elif args.domain:
        domains = [args.domain]
    else:
        parser.error("Must specify --domain or --all")

    # SMOTE configuration: Use SMOTE for imbalanced micro domains
    use_smote_dict = {
        # Macro domains - no SMOTE by default
        'has_suffix': args.use_smote,
        'has_mutation': args.use_smote,
        '3way': args.use_smote,
        # Micro domains - balanced or moderate imbalance
        'ablaut': args.use_smote,
        'medial_a': args.use_smote,
        # Micro domains - extreme imbalance (use SMOTE if flag provided OR by default)
        'final_a': True,  # 12.6% minority - always use SMOTE
        'final_vw': True,  # 4.8% minority - always use SMOTE
        'insert_c': True,  # 6.8% minority - always use SMOTE
        'templatic': args.use_smote,  # 18.1% minority - moderate
        '8way': args.use_smote  # Multiclass
    }

    # Train each domain
    start_time = datetime.now()
    summaries = []

    for domain in domains:
        domain_start = datetime.now()

        summary = train_domain(
            domain_name=domain,
            use_smote=use_smote_dict[domain],
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            verbose=args.verbose
        )
        summaries.append(summary)

        domain_elapsed = (datetime.now() - domain_start).total_seconds()
        print(f"\nDomain '{domain}' completed in {domain_elapsed/60:.1f} minutes")

    total_elapsed = (datetime.now() - start_time).total_seconds()

    # Final summary
    print("\n" + "="*80)
    print("ALL TRAINING COMPLETE")
    print("="*80)
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes")
    print(f"\nDomains trained: {', '.join(domains)}")

    print("\n=== Performance Summary ===")
    for domain, summary in zip(domains, summaries):
        row = summary.iloc[0]
        print(f"\n{domain}:")
        print(f"  Accuracy:  {row['accuracy_mean']:.3f} ± {row['accuracy_std']:.3f}")
        print(f"  Macro-F1:  {row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f}")
        print(f"  AUC-ROC:   {row['auc_roc_mean']:.3f} ± {row['auc_roc_std']:.3f}")

    print("\n" + "="*80)
    print("Next steps:")
    print("  1. Review results in experiments/results/lstm_baseline_*/")
    print("  2. Compare with existing ablation results")
    print("  3. Train combined model (Day 2)")
    print("="*80)


if __name__ == '__main__':
    main()
