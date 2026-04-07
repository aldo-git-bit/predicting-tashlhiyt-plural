"""
Consolidate Features Across Targets

Consolidates selected features from all 8 targets using union strategy.
For features selected by multiple targets, takes the maximum stability score.

Strategy:
- Union: Include a feature if selected by ANY target
- Max stability: For each feature, use the highest stability score across all targets
- Feature metadata: Track which targets selected each feature
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def load_target_results(results_dir, level='macro'):
    """
    Load results for all targets at a given level (macro or micro).

    Args:
        results_dir (Path): Directory containing results
        level (str): 'macro' or 'micro'

    Returns:
        list: List of (target_name, results_df) tuples
    """
    level_dir = Path(results_dir) / level
    if not level_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {level_dir}")

    results = []

    # Find all result files
    for results_file in sorted(level_dir.glob("*_results.csv")):
        target_name = results_file.stem.replace('_results', '')
        df = pd.read_csv(results_file)
        results.append((target_name, df))

    return results


def consolidate_features(all_target_results):
    """
    Consolidate features using union strategy with max stability.

    Args:
        all_target_results (list): List of (target_name, results_df) tuples

    Returns:
        DataFrame: Consolidated features with metadata
            - feature: N-gram
            - max_stability: Maximum stability score across targets
            - n_targets_selected: Number of targets that selected this feature
            - targets_selected: Comma-separated list of target names
            - mean_stability: Mean stability score across all targets
    """
    # Collect all features
    all_features = set()
    for target_name, results_df in all_target_results:
        all_features.update(results_df['feature'].tolist())

    all_features = sorted(all_features)

    # Build consolidated dataframe
    consolidated_data = []

    for feature in all_features:
        stability_scores = []
        targets_selected = []

        # Collect data from each target
        for target_name, results_df in all_target_results:
            row = results_df[results_df['feature'] == feature]

            if not row.empty:
                stability = row.iloc[0]['stability_score']
                selected = row.iloc[0]['selected']

                stability_scores.append(stability)

                if selected:
                    targets_selected.append(target_name)

        # Compute consolidated metrics
        consolidated_data.append({
            'feature': feature,
            'max_stability': max(stability_scores) if stability_scores else 0.0,
            'mean_stability': np.mean(stability_scores) if stability_scores else 0.0,
            'n_targets_selected': len(targets_selected),
            'targets_selected': ','.join(targets_selected) if targets_selected else ''
        })

    consolidated_df = pd.DataFrame(consolidated_data)

    # Sort by number of targets, then max stability
    consolidated_df = consolidated_df.sort_values(
        ['n_targets_selected', 'max_stability'],
        ascending=[False, False]
    )

    return consolidated_df


def get_final_feature_set(consolidated_df):
    """
    Extract final feature set (features selected by at least one target).

    Args:
        consolidated_df (DataFrame): Consolidated features

    Returns:
        list: Final feature names
    """
    selected = consolidated_df[consolidated_df['n_targets_selected'] > 0]
    return selected['feature'].tolist()


def consolidate_all_results(results_dir, output_dir=None):
    """
    Consolidate results from all targets (macro and micro).

    Args:
        results_dir (Path): Directory containing macro/ and micro/ subdirectories
        output_dir (Path): Directory to save consolidated results (optional)

    Returns:
        dict: Consolidated results for macro and micro levels
    """
    results_dir = Path(results_dir)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("CONSOLIDATING FEATURES ACROSS TARGETS")
    print(f"{'='*70}\n")

    consolidated = {}

    # Macro-level
    print("Macro-level targets:")
    macro_results = load_target_results(results_dir, 'macro')
    print(f"  Loaded {len(macro_results)} targets")

    macro_consolidated = consolidate_features(macro_results)
    macro_final = get_final_feature_set(macro_consolidated)

    print(f"  Total unique features: {len(macro_consolidated)}")
    print(f"  Selected features: {len(macro_final)}")
    print()

    consolidated['macro'] = {
        'consolidated_df': macro_consolidated,
        'final_features': macro_final
    }

    # Micro-level
    print("Micro-level targets:")
    micro_results = load_target_results(results_dir, 'micro')
    print(f"  Loaded {len(micro_results)} targets")

    micro_consolidated = consolidate_features(micro_results)
    micro_final = get_final_feature_set(micro_consolidated)

    print(f"  Total unique features: {len(micro_consolidated)}")
    print(f"  Selected features: {len(micro_final)}")
    print()

    consolidated['micro'] = {
        'consolidated_df': micro_consolidated,
        'final_features': micro_final
    }

    # Save results
    if output_dir is not None:
        # Macro
        macro_consolidated.to_csv(output_dir / 'macro_consolidated.csv', index=False)
        with open(output_dir / 'macro_final_features.txt', 'w') as f:
            for feature in macro_final:
                f.write(f"{feature}\n")

        # Micro
        micro_consolidated.to_csv(output_dir / 'micro_consolidated.csv', index=False)
        with open(output_dir / 'micro_final_features.txt', 'w') as f:
            for feature in micro_final:
                f.write(f"{feature}\n")

        # Summary
        summary = {
            'macro': {
                'n_total_features': len(macro_consolidated),
                'n_selected': len(macro_final),
                'selection_rate': len(macro_final) / len(macro_consolidated)
            },
            'micro': {
                'n_total_features': len(micro_consolidated),
                'n_selected': len(micro_final),
                'selection_rate': len(micro_final) / len(micro_consolidated)
            }
        }

        with open(output_dir / 'consolidation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Consolidated results saved to: {output_dir}")
        print()

    # Print summary statistics
    print(f"{'='*70}")
    print("CONSOLIDATION SUMMARY")
    print(f"{'='*70}\n")

    print("Macro-level:")
    print(f"  Features selected by 2 targets: {(macro_consolidated['n_targets_selected'] == 2).sum()}")
    print(f"  Features selected by 1 target: {(macro_consolidated['n_targets_selected'] == 1).sum()}")
    print(f"  Features not selected: {(macro_consolidated['n_targets_selected'] == 0).sum()}")
    print()

    print("Micro-level:")
    for n in range(6, 0, -1):
        count = (micro_consolidated['n_targets_selected'] == n).sum()
        if count > 0:
            print(f"  Features selected by {n} targets: {count}")
    print(f"  Features not selected: {(micro_consolidated['n_targets_selected'] == 0).sum()}")
    print()

    return consolidated


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python consolidate_features.py <results_dir>")
        print("\nExample:")
        print("  python consolidate_features.py ../../results/ngram_feature_selection/20251228_123456")
        sys.exit(1)

    results_dir = sys.argv[1]
    output_dir = Path(results_dir) / 'consolidated'

    try:
        consolidated = consolidate_all_results(results_dir, output_dir)

        print("✅ Consolidation complete")
        print(f"\nFinal feature sets:")
        print(f"  Macro: {len(consolidated['macro']['final_features'])} features")
        print(f"  Micro: {len(consolidated['micro']['final_features'])} features")

        # Show top 10 features by number of targets
        print(f"\nTop 10 macro features (by number of targets):")
        print(consolidated['macro']['consolidated_df'].head(10)[
            ['feature', 'n_targets_selected', 'max_stability']
        ])

        print(f"\nTop 10 micro features (by number of targets):")
        print(consolidated['micro']['consolidated_df'].head(10)[
            ['feature', 'n_targets_selected', 'max_stability']
        ])

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
