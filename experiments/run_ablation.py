"""
Ablation Study - All Domains

Compares 6 feature sets across 3 model types for any domain:
- Feature sets: N-grams only, Semantic, Morph, Phon, Morph+Phon, All
- Models: Logistic Regression, Random Forest, XGBoost
- Baselines: Majority Class, Stratified Random
- Total: 20 experiments per domain

Uses fixed hyperparameters (no tuning) to ensure observed performance
differences reflect the informational quality of linguistic features
rather than artifacts of optimization.

Usage:
    python experiments/run_ablation.py --domain has_suffix
    python experiments/run_ablation.py --domain medial_a
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_config,
    run_cross_validation,
    save_results,
    print_cv_summary
)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent

# Feature sets
FEATURE_SETS = {
    'ngrams_only': 'X_ngrams_only.csv',
    'semantic_only': 'X_semantic_only.csv',
    'morph_only': 'X_morph_only.csv',
    'phon_only': 'X_phon_only.csv',
    'morph_phon': 'X_morph_phon.csv',
    'all_features': 'X_all_features.csv'
}


def load_feature_set(domain: str, feature_set_name: str):
    """Load a specific feature set and target for a domain."""
    FEATURES_DIR = PROJECT_ROOT / 'features' / f'ablation_{domain}'

    X_path = FEATURES_DIR / FEATURE_SETS[feature_set_name]
    y_path = FEATURES_DIR / f'y_{domain}.csv'

    X = pd.read_csv(X_path, index_col=0)
    y = pd.read_csv(y_path, index_col=0).squeeze()

    return X, y


def run_majority_baseline(domain: str, config: Dict, use_smote: bool = False) -> Dict[str, Any]:
    """Run majority class baseline."""
    print(f"\n{'='*80}")
    print(f"BASELINE - Majority Class")
    print(f"{'='*80}")

    # Load data (use minimal feature set)
    X, y = load_feature_set(domain, 'semantic_only')
    print(f"  Data: {X.shape[0]} samples")

    # Print class distribution
    if y.dtype == 'object' or len(y.unique()) <= 10:
        print(f"  Class distribution:")
        for label, count in y.value_counts().sort_index().items():
            print(f"    {label}: {count} ({count/len(y):.1%})")

    # Create majority class classifier
    model = DummyClassifier(strategy='most_frequent', random_state=config['random_state'])

    # Run cross-validation (SMOTE doesn't apply to baselines)
    print(f"\n  Running {config['cv']['n_folds']}-fold CV...")
    results = run_cross_validation(
        model, X, y,
        n_folds=config['cv']['n_folds'],
        random_state=config['random_state'],
        use_smote=False  # Never use SMOTE for baselines
    )

    # Add metadata
    results['feature_set'] = 'baseline'
    results['model'] = 'majority_class'
    results['n_samples'] = len(X)
    results['n_features'] = 0

    # Print summary
    print_cv_summary(results, 'Majority Baseline', 'baseline')

    return results


def run_random_baseline(domain: str, config: Dict, use_smote: bool = False) -> Dict[str, Any]:
    """Run stratified random baseline."""
    print(f"\n{'='*80}")
    print(f"BASELINE - Stratified Random")
    print(f"{'='*80}")

    # Load data (use minimal feature set)
    X, y = load_feature_set(domain, 'semantic_only')
    print(f"  Data: {X.shape[0]} samples")

    # Print class distribution
    if y.dtype == 'object' or len(y.unique()) <= 10:
        print(f"  Class distribution:")
        for label, count in y.value_counts().sort_index().items():
            print(f"    {label}: {count} ({count/len(y):.1%})")

    # Create stratified random classifier
    model = DummyClassifier(strategy='stratified', random_state=config['random_state'])

    # Run cross-validation (SMOTE doesn't apply to baselines)
    print(f"\n  Running {config['cv']['n_folds']}-fold CV...")
    results = run_cross_validation(
        model, X, y,
        n_folds=config['cv']['n_folds'],
        random_state=config['random_state'],
        use_smote=False  # Never use SMOTE for baselines
    )

    # Add metadata
    results['feature_set'] = 'baseline'
    results['model'] = 'random_stratified'
    results['n_samples'] = len(X)
    results['n_features'] = 0

    # Print summary
    print_cv_summary(results, 'Random Baseline', 'baseline')

    return results


def run_logistic_regression(domain: str, feature_set: str, config: Dict, use_smote: bool = False) -> Dict[str, Any]:
    """Run Logistic Regression on a feature set."""
    print(f"\n{'='*80}")
    print(f"LOGISTIC REGRESSION - {feature_set}")
    print(f"{'='*80}")

    # Load data
    X, y = load_feature_set(domain, feature_set)
    print(f"  Data: {X.shape[0]} samples, {X.shape[1]} features")

    # Get fixed hyperparameters from config
    params = config['models']['logistic_regression']

    # Create model with fixed hyperparameters
    model = LogisticRegression(
        C=params['C'],
        penalty=params['penalty'],
        solver=params['solver'],
        max_iter=params['max_iter'],
        class_weight=params['class_weight'],
        random_state=config['random_state'],
        n_jobs=config['n_jobs']
    )

    # Run cross-validation
    print(f"\n  Running {config['cv']['n_folds']}-fold CV...")
    if use_smote:
        print(f"  SMOTE: Enabled (synthetic oversampling in each fold)")
    results = run_cross_validation(
        model, X, y,
        n_folds=config['cv']['n_folds'],
        random_state=config['random_state'],
        use_smote=use_smote
    )

    # Add metadata
    results['feature_set'] = feature_set
    results['model'] = 'logistic_regression'
    results['n_samples'] = len(X)
    results['n_features'] = X.shape[1]
    results['hyperparameters'] = {k: v for k, v in params.items() if k != 'name'}

    # Print summary
    print_cv_summary(results, 'Logistic Regression', feature_set)

    return results


def run_random_forest(domain: str, feature_set: str, config: Dict, use_smote: bool = False) -> Dict[str, Any]:
    """Run Random Forest on a feature set."""
    print(f"\n{'='*80}")
    print(f"RANDOM FOREST - {feature_set}")
    print(f"{'='*80}")

    # Load data
    X, y = load_feature_set(domain, feature_set)
    print(f"  Data: {X.shape[0]} samples, {X.shape[1]} features")

    # Get fixed hyperparameters from config
    params = config['models']['random_forest']

    # Create model with fixed hyperparameters
    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features=params['max_features'],
        class_weight=params['class_weight'],
        random_state=config['random_state'],
        n_jobs=config['n_jobs']
    )

    # Run cross-validation
    print(f"\n  Running {config['cv']['n_folds']}-fold CV...")
    if use_smote:
        print(f"  SMOTE: Enabled (synthetic oversampling in each fold)")
    results = run_cross_validation(
        model, X, y,
        n_folds=config['cv']['n_folds'],
        random_state=config['random_state'],
        use_smote=use_smote
    )

    # Add metadata
    results['feature_set'] = feature_set
    results['model'] = 'random_forest'
    results['n_samples'] = len(X)
    results['n_features'] = X.shape[1]
    results['hyperparameters'] = {k: v for k, v in params.items() if k != 'name'}

    # Print summary
    print_cv_summary(results, 'Random Forest', feature_set)

    return results


def run_xgboost(domain: str, feature_set: str, config: Dict, use_smote: bool = False) -> Dict[str, Any]:
    """Run XGBoost on a feature set."""
    print(f"\n{'='*80}")
    print(f"XGBOOST - {feature_set}")
    print(f"{'='*80}")

    # Load data
    X, y = load_feature_set(domain, feature_set)
    print(f"  Data: {X.shape[0]} samples, {X.shape[1]} features")

    # XGBoost requires numeric labels
    label_encoder = None
    if y.dtype == 'object' or isinstance(y.iloc[0], str):
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), index=y.index)
        print(f"  Encoded labels: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

    # Compute scale_pos_weight for binary classification
    if len(y.unique()) == 2:
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        print(f"  scale_pos_weight: {scale_pos_weight:.2f}")
    else:
        scale_pos_weight = 1.0

    # Get fixed hyperparameters from config
    params = config['models']['xgboost']

    # Create model with fixed hyperparameters
    model = XGBClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        min_child_weight=params['min_child_weight'],
        gamma=params['gamma'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        scale_pos_weight=scale_pos_weight,
        random_state=config['random_state'],
        n_jobs=config['n_jobs'],
        eval_metric='logloss'
    )

    # Run cross-validation
    print(f"\n  Running {config['cv']['n_folds']}-fold CV...")
    if use_smote:
        print(f"  SMOTE: Enabled (synthetic oversampling in each fold)")
    results = run_cross_validation(
        model, X, y,
        n_folds=config['cv']['n_folds'],
        random_state=config['random_state'],
        use_smote=use_smote
    )

    # Add metadata
    results['feature_set'] = feature_set
    results['model'] = 'xgboost'
    results['n_samples'] = len(X)
    results['n_features'] = X.shape[1]
    results['hyperparameters'] = {k: v for k, v in params.items() if k != 'name'}
    if label_encoder:
        results['label_encoding'] = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

    # Print summary
    print_cv_summary(results, 'XGBoost', feature_set)

    return results


def main():
    parser = argparse.ArgumentParser(description='Run ablation study for any domain')
    parser.add_argument('--domain', type=str, required=True,
                       help='Domain name (e.g., has_suffix, medial_a, 3way)')
    parser.add_argument('--use-smote', action='store_true',
                       help='Apply SMOTE to training data (recommended for imbalanced classes)')

    args = parser.parse_args()
    domain = args.domain
    use_smote = args.use_smote

    print("="*80)
    print(f"ABLATION STUDY: {domain}")
    print("Fixed hyperparameters - no tuning")
    if use_smote:
        print("SMOTE: ENABLED (synthetic oversampling of minority class)")
    print("="*80)
    print("\nFeature sets:")
    for i, (name, file) in enumerate(FEATURE_SETS.items(), 1):
        print(f"  {i}. {name}: {file}")

    print("\nModels:")
    print("  1. Logistic Regression")
    print("  2. Random Forest")
    print("  3. XGBoost")

    print("\nBaselines:")
    print("  1. Majority Class")
    print("  2. Stratified Random")

    print(f"\nTotal experiments: {len(FEATURE_SETS)} × 3 + 2 baselines = {len(FEATURE_SETS) * 3 + 2}")

    # Load config
    config = load_config()
    print(f"\nUsing {config['cv']['n_folds']}-fold cross-validation")
    print(f"Random state: {config['random_state']}")
    print(f"Parallel jobs: {config['n_jobs']}")

    # Create results directory
    RESULTS_DIR = PROJECT_ROOT / 'experiments' / 'results' / f'ablation_{domain}'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run experiments
    all_results = []
    total = len(FEATURE_SETS) * 3 + 2  # +2 for baselines
    completed = 0
    start_time = datetime.now()

    # Run baselines first
    try:
        results = run_majority_baseline(domain, config, use_smote)
        all_results.append(results)
        completed += 1
        print(f"\n  Progress: {completed}/{total} experiments ({completed/total*100:.1f}%)")
    except Exception as e:
        print(f"\n  ✗ ERROR in Majority Baseline: {e}")

    try:
        results = run_random_baseline(domain, config, use_smote)
        all_results.append(results)
        completed += 1
        print(f"\n  Progress: {completed}/{total} experiments ({completed/total*100:.1f}%)")
    except Exception as e:
        print(f"\n  ✗ ERROR in Random Baseline: {e}")

    # Run feature-based models
    for feature_set in FEATURE_SETS.keys():
        # Logistic Regression
        try:
            exp_start = datetime.now()
            results = run_logistic_regression(domain, feature_set, config, use_smote)
            save_results(results, f'ablation_{domain}',
                        f'logistic_regression_{feature_set}',
                        datetime.now().strftime('%Y%m%d_%H%M%S'))
            all_results.append(results)
            completed += 1
            elapsed = (datetime.now() - exp_start).total_seconds()
            total_elapsed = (datetime.now() - start_time).total_seconds()
            print(f"\n  ✓ Completed in {elapsed:.1f}s")
            print(f"  Progress: {completed}/{total} experiments ({completed/total*100:.1f}%)")
            print(f"  Total time elapsed: {total_elapsed/60:.1f} minutes")
        except Exception as e:
            print(f"\n  ✗ ERROR in Logistic Regression - {feature_set}: {e}")

        # Random Forest
        try:
            exp_start = datetime.now()
            results = run_random_forest(domain, feature_set, config, use_smote)
            save_results(results, f'ablation_{domain}',
                        f'random_forest_{feature_set}',
                        datetime.now().strftime('%Y%m%d_%H%M%S'))
            all_results.append(results)
            completed += 1
            elapsed = (datetime.now() - exp_start).total_seconds()
            total_elapsed = (datetime.now() - start_time).total_seconds()
            print(f"\n  ✓ Completed in {elapsed:.1f}s")
            print(f"  Progress: {completed}/{total} experiments ({completed/total*100:.1f}%)")
            print(f"  Total time elapsed: {total_elapsed/60:.1f} minutes")
        except Exception as e:
            print(f"\n  ✗ ERROR in Random Forest - {feature_set}: {e}")

        # XGBoost
        try:
            exp_start = datetime.now()
            results = run_xgboost(domain, feature_set, config, use_smote)
            save_results(results, f'ablation_{domain}',
                        f'xgboost_{feature_set}',
                        datetime.now().strftime('%Y%m%d_%H%M%S'))
            all_results.append(results)
            completed += 1
            elapsed = (datetime.now() - exp_start).total_seconds()
            total_elapsed = (datetime.now() - start_time).total_seconds()
            print(f"\n  ✓ Completed in {elapsed:.1f}s")
            print(f"  Progress: {completed}/{total} experiments ({completed/total*100:.1f}%)")
            print(f"  Total time elapsed: {total_elapsed/60:.1f} minutes")
        except Exception as e:
            print(f"\n  ✗ ERROR in XGBoost - {feature_set}: {e}")

    total_time = (datetime.now() - start_time).total_seconds()
    print("\n" + "="*80)
    print(f"ABLATION STUDY COMPLETE: {domain}")
    print("="*80)
    print(f"  Completed: {completed}/{total} experiments")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Average per experiment: {total_time/completed:.1f} seconds")

    # Save summary
    if all_results:
        summary_df = pd.DataFrame([{
            'feature_set': r['feature_set'],
            'model': r['model'],
            'n_features': r['n_features'],
            'accuracy_mean': r['overall_metrics']['accuracy_mean'],
            'accuracy_std': r['overall_metrics']['accuracy_std'],
            'macro_f1_mean': r['overall_metrics']['macro_f1_mean'],
            'macro_f1_std': r['overall_metrics']['macro_f1_std'],
            'auc_roc_mean': r['overall_metrics'].get('auc_roc_mean', 0),
            'auc_roc_std': r['overall_metrics'].get('auc_roc_std', 0),
            'precision_mean': r['overall_metrics'].get('precision_mean', r['overall_metrics'].get('precision_macro_mean', 0)),
            'precision_std': r['overall_metrics'].get('precision_std', r['overall_metrics'].get('precision_macro_std', 0)),
            'recall_mean': r['overall_metrics'].get('recall_mean', r['overall_metrics'].get('recall_macro_mean', 0)),
            'recall_std': r['overall_metrics'].get('recall_std', r['overall_metrics'].get('recall_macro_std', 0))
        } for r in all_results])

        summary_path = RESULTS_DIR / f'ablation_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved: {summary_path}")


if __name__ == '__main__':
    main()
