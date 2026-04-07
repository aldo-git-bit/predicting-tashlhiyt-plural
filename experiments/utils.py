"""
Utility functions for ML experiments.

Provides functions for:
- Loading features and targets
- Cross-validation
- Metric computation
- Result storage
"""

import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from typing import Tuple, Dict, List, Any
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from imblearn.over_sampling import SMOTE


# Paths
PROJECT_ROOT = Path(__file__).parent.parent
FEATURES_DIR = PROJECT_ROOT / "features"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
MODELS_DIR = PROJECT_ROOT / "experiments" / "models"


def load_config(config_path: str = None) -> Dict:
    """
    Load experiment configuration from YAML file.

    Args:
        config_path: Path to config file (default: experiments/config.yaml)

    Returns:
        Dictionary with configuration
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "experiments" / "config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def load_features_and_target(domain: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load feature matrix and target variable for a domain.

    Args:
        domain: Model domain (e.g., 'macro_has_suffix', 'micro_ablaut')

    Returns:
        Tuple of (X, y)
    """
    X_path = FEATURES_DIR / f"X_{domain}.csv"
    y_path = FEATURES_DIR / f"y_{domain}.csv"

    # Load features (recordID is index)
    X = pd.read_csv(X_path, index_col=0)

    # Load target
    y = pd.read_csv(y_path, index_col=0).squeeze()

    # Ensure alignment
    assert X.index.equals(y.index), f"Index mismatch for {domain}"

    return X, y


def load_metadata(domain: str) -> Dict:
    """
    Load feature metadata for a domain.

    Args:
        domain: Model domain

    Returns:
        Metadata dictionary
    """
    meta_path = FEATURES_DIR / f"feature_metadata_{domain}.json"

    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    return metadata


def get_feature_groups(X: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identify feature groups by prefix.

    Args:
        X: Feature matrix

    Returns:
        Dictionary mapping group name to list of feature names
    """
    groups = {
        'morphological': [col for col in X.columns if col.startswith('m_')],
        'semantic': [col for col in X.columns if col.startswith('s_')],
        'phonological': [col for col in X.columns if col.startswith('p_')],
        'ngrams': [col for col in X.columns if not any(col.startswith(p) for p in ['m_', 's_', 'p_'])]
    }

    return groups


def filter_features_by_groups(X: pd.DataFrame, include_groups: List[str]) -> pd.DataFrame:
    """
    Filter feature matrix to include only specified groups.

    Args:
        X: Full feature matrix
        include_groups: List of group names to include

    Returns:
        Filtered feature matrix
    """
    groups = get_feature_groups(X)

    selected_cols = []
    for group in include_groups:
        selected_cols.extend(groups.get(group, []))

    return X[selected_cols]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None,
                   is_binary: bool = True) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for AUC-ROC)
        is_binary: Whether this is binary classification

    Returns:
        Dictionary of metric name -> value
    """
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    # AUC-ROC (if probabilities provided)
    if y_proba is not None:
        try:
            if is_binary:
                # For binary: use probabilities of positive class
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
            else:
                # For multiclass: use OVR strategy
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        except ValueError:
            # In case of issues (e.g., only one class in y_true)
            metrics['auc_roc'] = np.nan

    # For binary classification
    if is_binary:
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)  # Positive class only
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)  # Macro-averaged across both classes
    else:
        # For multiclass
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['macro_f1'] = metrics['f1_macro']

        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return metrics


def run_cross_validation(model, X: pd.DataFrame, y: pd.Series, n_folds: int = 10,
                        random_state: int = 42, use_smote: bool = False) -> Dict[str, Any]:
    """
    Run stratified k-fold cross-validation with optional SMOTE.

    Args:
        model: Sklearn-compatible model
        X: Feature matrix
        y: Target variable
        n_folds: Number of CV folds
        random_state: Random seed
        use_smote: Whether to apply SMOTE to training data (default: False)

    Returns:
        Dictionary with CV results
    """
    # Determine if binary or multiclass
    is_binary = len(y.unique()) == 2

    # Set up cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Storage for results
    fold_metrics = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    all_record_ids = []

    # Run CV
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Get record IDs for validation set
        record_ids_val = X_val.index.tolist()

        # Apply SMOTE to training data if requested
        if use_smote:
            # Determine minority class size
            class_counts = y_train.value_counts()
            min_samples = class_counts.min()

            # Only apply SMOTE if minority class has at least 2 samples
            # (SMOTE requires k_neighbors >= 1, default is 5)
            if min_samples >= 2:
                # Adjust k_neighbors based on minority class size
                k_neighbors = min(5, min_samples - 1)

                try:
                    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                except Exception as e:
                    # If SMOTE fails, continue without it for this fold
                    print(f"    Warning: SMOTE failed for fold {fold_idx}: {e}")

        # Train model
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_val)

        # Get probabilities for AUC-ROC
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_val)
                if is_binary:
                    # For binary classification, use probability of positive class (class 1)
                    y_proba_pos = y_proba[:, 1]
                else:
                    # For multiclass, use all probabilities
                    y_proba_pos = y_proba
            else:
                # Models without predict_proba (e.g., some linear models)
                y_proba_pos = None
        except:
            y_proba_pos = None

        # Compute metrics for this fold
        metrics = compute_metrics(y_val, y_pred, y_proba=y_proba_pos, is_binary=is_binary)
        metrics['fold'] = fold_idx
        fold_metrics.append(metrics)

        # Store predictions, probabilities, and record IDs
        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_record_ids.extend(record_ids_val)

        # Store probabilities (handle both binary and multiclass)
        if y_proba is not None:
            # Store full probability distributions
            all_y_proba.extend(y_proba.tolist())
        else:
            # No probabilities available - store None placeholders
            all_y_proba.extend([None] * len(y_val))

    # Convert to DataFrame
    fold_results = pd.DataFrame(fold_metrics)

    # Compute overall metrics
    overall_metrics = {}
    for metric in fold_results.columns:
        if metric != 'fold':
            overall_metrics[f'{metric}_mean'] = fold_results[metric].mean()
            overall_metrics[f'{metric}_std'] = fold_results[metric].std()

    # Overall confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)

    # Classification report
    report = classification_report(all_y_true, all_y_pred, output_dict=True, zero_division=0)

    return {
        'fold_results': fold_results,
        'overall_metrics': overall_metrics,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'predictions': {
            'y_true': all_y_true,
            'y_pred': all_y_pred,
            'y_proba': all_y_proba,
            'record_ids': all_record_ids
        }
    }


def save_results(results: Dict, domain: str, model_name: str, timestamp: str = None):
    """
    Save experiment results to JSON file.

    Args:
        results: Results dictionary
        domain: Model domain
        model_name: Name of the model
        timestamp: Optional timestamp (default: current time)
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create domain subdirectory
    domain_dir = RESULTS_DIR / domain
    domain_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    filename = f"{model_name}_{timestamp}.json"
    filepath = domain_dir / filename

    # Convert DataFrame to dict if needed
    if 'fold_results' in results and isinstance(results['fold_results'], pd.DataFrame):
        results['fold_results'] = results['fold_results'].to_dict(orient='records')

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved: {filepath}")

    return filepath


def load_results(domain: str, model_name: str, timestamp: str = None) -> Dict:
    """
    Load experiment results from JSON file.

    Args:
        domain: Model domain
        model_name: Name of the model
        timestamp: Optional timestamp (if None, loads most recent)

    Returns:
        Results dictionary
    """
    domain_dir = RESULTS_DIR / domain

    if timestamp is None:
        # Find most recent file for this model
        files = list(domain_dir.glob(f"{model_name}_*.json"))
        if not files:
            raise FileNotFoundError(f"No results found for {domain}/{model_name}")
        filepath = max(files, key=lambda p: p.stat().st_mtime)
    else:
        filepath = domain_dir / f"{model_name}_{timestamp}.json"

    with open(filepath, 'r') as f:
        results = json.load(f)

    return results


def print_cv_summary(results: Dict, model_name: str, domain: str):
    """
    Print formatted summary of CV results.

    Args:
        results: Results dictionary from run_cross_validation
        model_name: Name of the model
        domain: Model domain
    """
    print(f"\n{'='*80}")
    print(f"{model_name} - {domain}")
    print(f"{'='*80}")

    metrics = results['overall_metrics']

    # Print key metrics
    print(f"\nCross-Validation Results (10-fold):")
    print(f"  Macro-F1:  {metrics['macro_f1_mean']:.4f} ± {metrics['macro_f1_std']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}")
    if 'auc_roc_mean' in metrics:
        print(f"  AUC-ROC:   {metrics['auc_roc_mean']:.4f} ± {metrics['auc_roc_std']:.4f}")

    if 'precision_macro_mean' in metrics:
        print(f"  Precision: {metrics['precision_macro_mean']:.4f} ± {metrics['precision_macro_std']:.4f}")
        print(f"  Recall:    {metrics['recall_macro_mean']:.4f} ± {metrics['recall_macro_std']:.4f}")
    elif 'precision_mean' in metrics:
        print(f"  Precision: {metrics['precision_mean']:.4f} ± {metrics['precision_std']:.4f}")
        print(f"  Recall:    {metrics['recall_mean']:.4f} ± {metrics['recall_std']:.4f}")

    # Confusion matrix
    print(f"\nOverall Confusion Matrix:")
    cm = np.array(results['confusion_matrix'])
    print(cm)

    print()


if __name__ == '__main__':
    # Test utilities
    print("Testing utility functions...\n")

    # Load config
    config = load_config()
    print(f"✓ Config loaded: {len(config)} sections")

    # Load a sample domain
    domain = 'macro_has_suffix'
    X, y = load_features_and_target(domain)
    print(f"✓ Loaded {domain}: X={X.shape}, y={y.shape}")

    # Get feature groups
    groups = get_feature_groups(X)
    for name, cols in groups.items():
        print(f"  {name}: {len(cols)} features")

    # Load metadata
    meta = load_metadata(domain)
    print(f"✓ Metadata loaded: {meta['n_samples']} samples, {meta['n_features']} features")

    print("\n✓ All utility functions working correctly")
