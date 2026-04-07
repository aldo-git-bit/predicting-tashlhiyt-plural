"""
Run Feature Selection for All 8 Targets

Orchestrates LASSO Stability Selection for all target variables:
- 2 macro-level targets (n=1,185)
- 6 micro-level targets (n=562)

Outputs:
- Individual results for each target
- Saved results in results/ directory
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

from feature_matrix_builder import build_feature_matrix_from_dataset
from target_preparation import prepare_all_targets
from lasso_stability import LASSOStabilitySelector


def run_selection_for_target(
    X, y, target_name, n_iterations=100, output_dir=None
):
    """
    Run feature selection for a single target.

    Args:
        X (DataFrame): Standardized feature matrix
        y (Series): Binary target variable
        target_name (str): Name of target (e.g., "y_macro_suffix")
        n_iterations (int): Number of bootstrap iterations
        output_dir (Path): Directory to save results (optional)

    Returns:
        dict: Results including selected features and diagnostics
    """
    print(f"\n{'='*70}")
    print(f"TARGET: {target_name}")
    print(f"{'='*70}\n")

    # Check target distribution
    n_positive = y.sum()
    n_total = len(y)
    pos_rate = n_positive / n_total * 100

    print(f"Dataset:")
    print(f"  Samples: {n_total}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Positive class: {n_positive} ({pos_rate:.1f}%)")
    print()

    # Run stability selection
    selector = LASSOStabilitySelector(
        n_iterations=n_iterations,
        stability_threshold=0.5,
        l1_ratio=0.95,
        cv_folds=5,
        random_state=42
    )

    selector.fit(X, y)

    # Get results
    results_df = selector.get_results()
    selected_features = selector.get_selected_features()

    # Summary statistics
    summary = {
        'target_name': target_name,
        'n_samples': n_total,
        'n_features': X.shape[1],
        'n_selected': len(selected_features),
        'selection_rate': len(selected_features) / X.shape[1],
        'n_positive': int(n_positive),
        'positive_rate': pos_rate / 100,
        'n_iterations': n_iterations,
        'timestamp': datetime.now().isoformat()
    }

    # Save results if output directory provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        results_file = output_dir / f"{target_name}_results.csv"
        results_df.to_csv(results_file, index=False)

        # Save selected features
        selected_file = output_dir / f"{target_name}_selected.txt"
        with open(selected_file, 'w') as f:
            for feature in selected_features:
                f.write(f"{feature}\n")

        # Save summary
        summary_file = output_dir / f"{target_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved:")
        print(f"  {results_file}")
        print(f"  {selected_file}")
        print(f"  {summary_file}")

    return {
        'target_name': target_name,
        'results_df': results_df,
        'selected_features': selected_features,
        'summary': summary
    }


def run_all_targets(
    df,
    n_iterations=100,
    output_dir='results/ngram_feature_selection',
    max_n=3
):
    """
    Run feature selection for all 8 targets.

    Args:
        df (DataFrame): Full dataset
        n_iterations (int): Number of bootstrap iterations
        output_dir (str): Directory to save results
        max_n (int): Maximum n-gram size

    Returns:
        dict: All results organized by target level (macro/micro)
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("N-GRAM FEATURE SELECTION - ALL TARGETS")
    print(f"{'='*70}\n")
    print(f"Configuration:")
    print(f"  Bootstrap iterations: {n_iterations}")
    print(f"  Stability threshold: 0.5")
    print(f"  L1 ratio: 0.95")
    print(f"  Max n-gram size: {max_n}")
    print(f"  Output directory: {output_dir}")
    print()

    all_results = {
        'macro': [],
        'micro': [],
        'metadata': {
            'timestamp': timestamp,
            'n_iterations': n_iterations,
            'max_n': max_n
        }
    }

    # Prepare targets
    macro_targets, micro_targets, metadata = prepare_all_targets(df)

    print(f"Target variables prepared:")
    print(f"  Macro-level: {len(macro_targets)} samples")
    print(f"  Micro-level: {len(micro_targets)} samples")
    print()

    # MACRO-LEVEL TARGETS
    print(f"{'='*70}")
    print("MACRO-LEVEL TARGETS (n=1,185)")
    print(f"{'='*70}")

    # Filter dataset to macro samples
    macro_indices = macro_targets.index
    df_macro = df.loc[macro_indices]

    # Build feature matrix for macro
    X_macro, scaler_macro, meta_macro = build_feature_matrix_from_dataset(
        df_macro,
        max_n=max_n,
        standardize=True
    )

    print(f"\nFeature matrix: {X_macro.shape[0]} samples × {X_macro.shape[1]} features")

    # Run for each macro target
    for target_col in ['y_macro_suffix', 'y_macro_mutated']:
        y = macro_targets[target_col]

        result = run_selection_for_target(
            X_macro, y, target_col,
            n_iterations=n_iterations,
            output_dir=output_dir / 'macro'
        )

        all_results['macro'].append(result)

    # MICRO-LEVEL TARGETS
    print(f"\n{'='*70}")
    print("MICRO-LEVEL TARGETS (n=562)")
    print(f"{'='*70}")

    # Filter dataset to micro samples
    micro_indices = micro_targets.index
    df_micro = df.loc[micro_indices]

    # Build feature matrix for micro
    X_micro, scaler_micro, meta_micro = build_feature_matrix_from_dataset(
        df_micro,
        max_n=max_n,
        standardize=True
    )

    print(f"\nFeature matrix: {X_micro.shape[0]} samples × {X_micro.shape[1]} features")

    # Run for each micro target
    micro_target_cols = [
        'y_micro_ablaut',
        'y_micro_templatic',
        'y_micro_medial_a',
        'y_micro_final_a',
        'y_micro_final_vw',
        'y_micro_insert_c'
    ]

    for target_col in micro_target_cols:
        y = micro_targets[target_col]

        result = run_selection_for_target(
            X_micro, y, target_col,
            n_iterations=n_iterations,
            output_dir=output_dir / 'micro'
        )

        all_results['micro'].append(result)

    # Save master summary
    master_summary = {
        'timestamp': timestamp,
        'n_iterations': n_iterations,
        'max_n': max_n,
        'macro_targets': [r['summary'] for r in all_results['macro']],
        'micro_targets': [r['summary'] for r in all_results['micro']]
    }

    summary_file = output_dir / 'master_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(master_summary, f, indent=2)

    print(f"\n{'='*70}")
    print("FEATURE SELECTION COMPLETE")
    print(f"{'='*70}\n")
    print(f"Results saved to: {output_dir}")
    print(f"Master summary: {summary_file}")
    print()

    # Summary table
    print("Summary of Selected Features:")
    print(f"\n{'Target':<25} {'Samples':<10} {'Features':<12} {'Selected':<12} {'Rate':<10}")
    print("-" * 70)

    for result in all_results['macro']:
        s = result['summary']
        print(f"{s['target_name']:<25} {s['n_samples']:<10} {s['n_features']:<12} "
              f"{s['n_selected']:<12} {s['selection_rate']*100:>6.1f}%")

    for result in all_results['micro']:
        s = result['summary']
        print(f"{s['target_name']:<25} {s['n_samples']:<10} {s['n_features']:<12} "
              f"{s['n_selected']:<12} {s['selection_rate']*100:>6.1f}%")

    return all_results


if __name__ == '__main__':
    import sys

    # Load dataset
    df = pd.read_csv('../../data/tash_nouns.csv')

    # Get parameters from command line
    n_iterations = 100
    if len(sys.argv) > 1:
        n_iterations = int(sys.argv[1])

    print(f"\nRunning feature selection with {n_iterations} bootstrap iterations...")
    print(f"This will take approximately {n_iterations * 8 * 2 / 60:.0f} minutes.\n")

    # Run all targets
    results = run_all_targets(
        df,
        n_iterations=n_iterations,
        output_dir='../../results/ngram_feature_selection'
    )

    print("\n✅ All feature selection complete")
