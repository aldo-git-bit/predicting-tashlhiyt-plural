"""
Cross-Validation of Selected Features

Validates predictive performance of selected features using:
- 5-fold stratified cross-validation
- Logistic Regression classifier
- Comparison: selected features vs full feature set

Metrics:
- Accuracy
- Precision, Recall, F1 (macro-averaged)
- ROC-AUC
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
import warnings


def cross_validate_features(X, y, cv_folds=5):
    """
    Run cross-validation on a feature set.

    Args:
        X (DataFrame): Feature matrix (standardized)
        y (Series): Binary target
        cv_folds (int): Number of cross-validation folds

    Returns:
        dict: Cross-validation results
    """
    # Define scoring metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='binary', zero_division=0),
        'recall': make_scorer(recall_score, average='binary', zero_division=0),
        'f1': make_scorer(f1_score, average='binary', zero_division=0),
        'roc_auc': make_scorer(roc_auc_score)
    }

    # Logistic Regression with cross-validation for lambda tuning
    model = LogisticRegressionCV(
        cv=3,  # Inner CV for lambda
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )

    # Outer cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )

    # Aggregate results
    results = {
        'accuracy': {
            'mean': cv_results['test_accuracy'].mean(),
            'std': cv_results['test_accuracy'].std(),
            'scores': cv_results['test_accuracy'].tolist()
        },
        'precision': {
            'mean': cv_results['test_precision'].mean(),
            'std': cv_results['test_precision'].std(),
            'scores': cv_results['test_precision'].tolist()
        },
        'recall': {
            'mean': cv_results['test_recall'].mean(),
            'std': cv_results['test_recall'].std(),
            'scores': cv_results['test_recall'].tolist()
        },
        'f1': {
            'mean': cv_results['test_f1'].mean(),
            'std': cv_results['test_f1'].std(),
            'scores': cv_results['test_f1'].tolist()
        },
        'roc_auc': {
            'mean': cv_results['test_roc_auc'].mean(),
            'std': cv_results['test_roc_auc'].std(),
            'scores': cv_results['test_roc_auc'].tolist()
        }
    }

    return results


def validate_target(
    X_full, X_selected, y, target_name, cv_folds=5
):
    """
    Validate selected features vs full feature set for one target.

    Args:
        X_full (DataFrame): Full feature matrix
        X_selected (DataFrame): Selected features only
        y (Series): Binary target
        target_name (str): Name of target
        cv_folds (int): CV folds

    Returns:
        dict: Validation results
    """
    print(f"\n{'='*70}")
    print(f"VALIDATING: {target_name}")
    print(f"{'='*70}\n")

    print(f"Full features: {X_full.shape[1]}")
    print(f"Selected features: {X_selected.shape[1]}")
    print(f"Reduction: {(1 - X_selected.shape[1] / X_full.shape[1]) * 100:.1f}%")
    print()

    # Validate full feature set
    print("Cross-validating full feature set...")
    results_full = cross_validate_features(X_full, y, cv_folds)

    # Validate selected features
    print("Cross-validating selected features...")
    results_selected = cross_validate_features(X_selected, y, cv_folds)

    # Comparison
    print(f"\n{'Metric':<15} {'Full':<20} {'Selected':<20} {'Diff':<10}")
    print("-" * 70)

    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        full_val = results_full[metric]['mean']
        selected_val = results_selected[metric]['mean']
        diff = selected_val - full_val

        print(f"{metric.capitalize():<15} {full_val:.3f} ± {results_full[metric]['std']:.3f}"
              f"    {selected_val:.3f} ± {results_selected[metric]['std']:.3f}"
              f"    {diff:+.3f}")

    return {
        'target_name': target_name,
        'n_features_full': X_full.shape[1],
        'n_features_selected': X_selected.shape[1],
        'reduction_pct': (1 - X_selected.shape[1] / X_full.shape[1]) * 100,
        'results_full': results_full,
        'results_selected': results_selected
    }


def validate_all_targets(df, consolidated_dir, results_dir, cv_folds=5):
    """
    Validate selected features for all targets.

    Args:
        df (DataFrame): Full dataset
        consolidated_dir (Path): Directory with consolidated results
        results_dir (Path): Directory with full results
        cv_folds (int): CV folds

    Returns:
        dict: All validation results
    """
    from feature_matrix_builder import build_feature_matrix_from_dataset
    from target_preparation import prepare_all_targets

    consolidated_dir = Path(consolidated_dir)
    results_dir = Path(results_dir)

    print(f"\n{'='*70}")
    print("CROSS-VALIDATION OF SELECTED FEATURES")
    print(f"{'='*70}\n")

    # Prepare targets
    macro_targets, micro_targets, metadata = prepare_all_targets(df)

    # Load consolidated features
    macro_features = pd.read_csv(consolidated_dir / 'macro_final_features.txt', header=None)[0].tolist()
    micro_features = pd.read_csv(consolidated_dir / 'micro_final_features.txt', header=None)[0].tolist()

    validation_results = {
        'macro': [],
        'micro': []
    }

    # MACRO VALIDATION
    print(f"{'='*70}")
    print("MACRO-LEVEL VALIDATION")
    print(f"{'='*70}")

    df_macro = df.loc[macro_targets.index]

    # Full feature matrix
    X_macro_full, _, _ = build_feature_matrix_from_dataset(df_macro, standardize=True)

    # Selected features only
    X_macro_selected = X_macro_full[macro_features]

    for target_col in ['y_macro_suffix', 'y_macro_mutated']:
        y = macro_targets[target_col]

        result = validate_target(
            X_macro_full, X_macro_selected, y, target_col, cv_folds
        )

        validation_results['macro'].append(result)

    # MICRO VALIDATION
    print(f"\n{'='*70}")
    print("MICRO-LEVEL VALIDATION")
    print(f"{'='*70}")

    df_micro = df.loc[micro_targets.index]

    # Full feature matrix
    X_micro_full, _, _ = build_feature_matrix_from_dataset(df_micro, standardize=True)

    # Selected features only
    X_micro_selected = X_micro_full[micro_features]

    micro_target_cols = [
        'y_macro_ablaut',
        'y_micro_templatic',
        'y_micro_medial_a',
        'y_micro_final_a',
        'y_micro_final_vw',
        'y_micro_insert_c'
    ]

    for target_col in micro_target_cols:
        if target_col not in micro_targets.columns:
            continue

        y = micro_targets[target_col]

        result = validate_target(
            X_micro_full, X_micro_selected, y, target_col, cv_folds
        )

        validation_results['micro'].append(result)

    # Save results
    output_file = consolidated_dir / 'validation_results.json'

    # Make results JSON serializable
    validation_json = {
        'macro': [
            {k: v for k, v in r.items() if k != 'results_full' and k != 'results_selected'}
            for r in validation_results['macro']
        ],
        'micro': [
            {k: v for k, v in r.items() if k != 'results_full' and k != 'results_selected'}
            for r in validation_results['micro']
        ]
    }

    # Add summary metrics
    for level in ['macro', 'micro']:
        for result in validation_results[level]:
            summary_key = f"{result['target_name']}_summary"
            validation_json[summary_key] = {
                'full': {
                    'accuracy': result['results_full']['accuracy']['mean'],
                    'f1': result['results_full']['f1']['mean'],
                    'roc_auc': result['results_full']['roc_auc']['mean']
                },
                'selected': {
                    'accuracy': result['results_selected']['accuracy']['mean'],
                    'f1': result['results_selected']['f1']['mean'],
                    'roc_auc': result['results_selected']['roc_auc']['mean']
                }
            }

    with open(output_file, 'w') as f:
        json.dump(validation_json, f, indent=2)

    print(f"\n✅ Validation results saved to: {output_file}")

    return validation_results


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python validate_features.py <results_timestamp>")
        print("\nExample:")
        print("  python validate_features.py 20251228_123456")
        sys.exit(1)

    timestamp = sys.argv[1]
    results_dir = Path(f"../../results/ngram_feature_selection/{timestamp}")
    consolidated_dir = results_dir / 'consolidated'

    if not consolidated_dir.exists():
        print(f"❌ Error: Consolidated results not found at {consolidated_dir}")
        print("Please run consolidate_features.py first.")
        sys.exit(1)

    try:
        df = pd.read_csv('../../data/tash_nouns.csv')

        results = validate_all_targets(
            df,
            consolidated_dir,
            results_dir,
            cv_folds=5
        )

        print("\n✅ Validation complete")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
