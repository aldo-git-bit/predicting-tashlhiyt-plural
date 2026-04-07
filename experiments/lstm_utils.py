#!/usr/bin/env python3
"""
Utility functions for LSTM training and evaluation.

Includes:
- Custom Keras metrics (Macro-F1, AUC-ROC)
- Data loading helpers
- Callbacks for early stopping
- Result saving functions
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report


class MacroF1(keras.metrics.Metric):
    """Custom Keras metric for macro-averaged F1 score."""

    def __init__(self, num_classes, name='macro_f1', **kwargs):
        super(MacroF1, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(name='tp', shape=(num_classes,), initializer='zeros')
        self.false_positives = self.add_weight(name='fp', shape=(num_classes,), initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', shape=(num_classes,), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        # Vectorized computation for all classes
        # Convert labels to one-hot
        y_true_onehot = tf.one_hot(y_true, depth=self.num_classes)
        y_pred_onehot = tf.one_hot(y_pred, depth=self.num_classes)

        # Compute TP, FP, FN for all classes at once
        tp = tf.reduce_sum(y_true_onehot * y_pred_onehot, axis=0)
        fp = tf.reduce_sum((1 - y_true_onehot) * y_pred_onehot, axis=0)
        fn = tf.reduce_sum(y_true_onehot * (1 - y_pred_onehot), axis=0)

        # Update state variables
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + keras.backend.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + keras.backend.epsilon())
        return tf.reduce_mean(f1)

    def reset_state(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))


def load_lstm_data(data_path):
    """
    Load preprocessed LSTM data.

    Args:
        data_path: Path to .npz file

    Returns:
        dict: Data dictionary with X, y, themes, label_map, etc.
    """
    data = np.load(data_path, allow_pickle=True)

    return {
        'X': data['X'],
        'y': data['y'],
        'themes': data['themes'],
        'original_indices': data['original_indices'],
        'label_map': json.loads(str(data['label_map'])),
        'n_classes': int(data['n_classes']),
        'vocab_size': int(data['vocab_size']),
        'max_len': int(data['max_len']),
        'domain_name': str(data['domain_name']),
        'folds': data['folds']
    }


def compute_metrics(y_true, y_pred, y_pred_proba):
    """
    Compute classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (n_samples, n_classes)

    Returns:
        dict: Metrics dictionary
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
    }

    # AUC-ROC (handle binary and multiclass)
    if y_pred_proba.shape[1] == 2:
        # Binary classification: use positive class probability
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
    else:
        # Multiclass: use ovr strategy
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
        except ValueError:
            # If some classes have no samples, AUC undefined
            metrics['auc_roc'] = np.nan

    return metrics


def save_fold_results(results_dir, fold_idx, y_true, y_pred, y_pred_proba, themes, history, metrics, record_ids=None):
    """
    Save results for a single fold.

    Args:
        results_dir: Directory to save results
        fold_idx: Fold index
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        themes: Original theme strings
        history: Training history object
        metrics: Computed metrics dictionary
        record_ids: Record IDs for validation samples (optional)
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions and probabilities
    pred_path = results_dir / f'predictions_fold_{fold_idx}.npz'
    save_data = {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'themes': themes
    }
    if record_ids is not None:
        save_data['record_ids'] = record_ids

    np.savez_compressed(pred_path, **save_data)

    # Save metrics as JSON
    metrics_path = results_dir / f'fold_{fold_idx}.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'fold': fold_idx,
            'metrics': metrics,
            'n_samples': len(y_true),
            'class_distribution': {
                int(k): int(v) for k, v in zip(*np.unique(y_true, return_counts=True))
            }
        }, f, indent=2)

    # Save training history
    history_path = results_dir / f'history_fold_{fold_idx}.json'
    with open(history_path, 'w') as f:
        json.dump({
            'history': {k: [float(x) for x in v] for k, v in history.history.items()},
            'epochs': len(history.history['loss'])
        }, f, indent=2)

    print(f"  Saved fold {fold_idx} results to {results_dir}")


def create_summary_csv(results_dir, domain_name, model_type):
    """
    Aggregate fold results into summary CSV.

    Args:
        results_dir: Directory with fold results
        domain_name: Domain name
        model_type: 'lstm' or 'lstm_combined'

    Returns:
        pd.DataFrame: Summary statistics
    """
    import pandas as pd

    results_dir = Path(results_dir)

    # Load all fold results
    fold_metrics = []
    for fold_file in sorted(results_dir.glob('fold_*.json')):
        with open(fold_file) as f:
            data = json.load(f)
            fold_metrics.append(data['metrics'])

    # Convert to DataFrame
    df = pd.DataFrame(fold_metrics)

    # Compute mean and std
    summary = {
        'domain': domain_name,
        'model': model_type,
        'accuracy_mean': df['accuracy'].mean(),
        'accuracy_std': df['accuracy'].std(),
        'macro_f1_mean': df['macro_f1'].mean(),
        'macro_f1_std': df['macro_f1'].std(),
        'auc_roc_mean': df['auc_roc'].mean(),
        'auc_roc_std': df['auc_roc'].std(),
    }

    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_path = results_dir / f'{model_type}_summary.csv'
    summary_df.to_csv(summary_path, index=False)

    print(f"\n=== Summary Statistics ===")
    print(f"Accuracy: {summary['accuracy_mean']:.3f} ± {summary['accuracy_std']:.3f}")
    print(f"Macro-F1: {summary['macro_f1_mean']:.3f} ± {summary['macro_f1_std']:.3f}")
    print(f"AUC-ROC:  {summary['auc_roc_mean']:.3f} ± {summary['auc_roc_std']:.3f}")
    print(f"Saved summary to: {summary_path}")

    return summary_df


def apply_smote_if_needed(X_train, y_train, min_samples_threshold=50):
    """
    Apply SMOTE to training data if minority class is too small.

    Args:
        X_train: Training sequences (2D array)
        y_train: Training labels
        min_samples_threshold: Apply SMOTE if minority class < this

    Returns:
        tuple: (X_train_resampled, y_train_resampled)
    """
    from collections import Counter

    class_counts = Counter(y_train)
    min_class_count = min(class_counts.values())

    print(f"    Class distribution before SMOTE: {dict(class_counts)}")

    if min_class_count < min_samples_threshold:
        print(f"    Applying SMOTE (minority class n={min_class_count})")

        try:
            from imblearn.over_sampling import SMOTE

            # Determine k_neighbors (must be < minority samples)
            k_neighbors = min(5, min_class_count - 1)
            if k_neighbors < 1:
                print(f"    WARNING: Minority class too small ({min_class_count}), skipping SMOTE")
                return X_train, y_train

            # Flatten sequences for SMOTE (SMOTE expects 2D)
            # X_train is already (n_samples, seq_len)
            smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            resampled_counts = Counter(y_train_resampled)
            print(f"    Class distribution after SMOTE: {dict(resampled_counts)}")

            return X_train_resampled, y_train_resampled

        except Exception as e:
            print(f"    WARNING: SMOTE failed: {e}")
            return X_train, y_train
    else:
        print(f"    Skipping SMOTE (minority class n={min_class_count} >= {min_samples_threshold})")
        return X_train, y_train
