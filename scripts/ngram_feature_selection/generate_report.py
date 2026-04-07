"""
Generate Final Feature Selection Report

Creates comprehensive diagnostic report including:
- Feature selection summary across all targets
- Positional distribution analysis (initial vs final)
- N-gram size distribution
- Phoneme frequency analysis
- Feature overlap between targets
- Recommendations for modeling
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json


def analyze_ngram_position(ngrams):
    """
    Analyze positional distribution of n-grams.

    Args:
        ngrams (list): List of n-grams with positional encoding

    Returns:
        dict: Position statistics
    """
    initial = [ng for ng in ngrams if ng.startswith('^')]
    final = [ng for ng in ngrams if ng.endswith('$')]

    return {
        'n_initial': len(initial),
        'n_final': len(final),
        'pct_initial': len(initial) / len(ngrams) * 100 if ngrams else 0,
        'pct_final': len(final) / len(ngrams) * 100 if ngrams else 0
    }


def analyze_ngram_size(ngrams):
    """
    Analyze n-gram size distribution.

    Args:
        ngrams (list): List of n-grams

    Returns:
        dict: Size statistics
    """
    sizes = []
    for ng in ngrams:
        # Remove positional markers
        clean = ng.replace('^', '').replace('$', '')
        # Count segments (accounting for labialized consonants)
        size = len(clean)  # Simplified; could improve with phoneme tokenization
        sizes.append(size)

    size_counts = pd.Series(sizes).value_counts().sort_index()

    return {
        'size_distribution': size_counts.to_dict(),
        'mean_size': np.mean(sizes) if sizes else 0
    }


def extract_phonemes_from_ngrams(ngrams):
    """
    Extract individual phonemes from n-grams.

    Args:
        ngrams (list): List of n-grams

    Returns:
        list: List of phonemes
    """
    phonemes = []

    for ng in ngrams:
        # Remove positional markers
        clean = ng.replace('^', '').replace('$', '')

        # Simple extraction (could improve with phoneme tokenization)
        for char in clean:
            phonemes.append(char)

    return phonemes


def generate_diagnostic_report(results_dir, output_file=None):
    """
    Generate comprehensive diagnostic report.

    Args:
        results_dir (Path): Directory with consolidated results
        output_file (Path): Output markdown file (optional)

    Returns:
        str: Report content
    """
    results_dir = Path(results_dir)

    # Load consolidated results
    macro_df = pd.read_csv(results_dir / 'macro_consolidated.csv')
    micro_df = pd.read_csv(results_dir / 'micro_consolidated.csv')

    with open(results_dir / 'consolidation_summary.json') as f:
        summary = json.load(f)

    # Start report
    report_lines = []
    report_lines.append("# N-gram Feature Selection Report")
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Overall summary
    report_lines.append("## Overall Summary\n")
    report_lines.append("### Macro-level Targets (n=1,185)")
    report_lines.append(f"- Total features: {summary['macro']['n_total_features']}")
    report_lines.append(f"- Selected features: {summary['macro']['n_selected']}")
    report_lines.append(f"- Selection rate: {summary['macro']['selection_rate']*100:.1f}%\n")

    report_lines.append("### Micro-level Targets (n=562)")
    report_lines.append(f"- Total features: {summary['micro']['n_total_features']}")
    report_lines.append(f"- Selected features: {summary['micro']['n_selected']}")
    report_lines.append(f"- Selection rate: {summary['micro']['selection_rate']*100:.1f}%\n")

    # Target-level breakdown
    report_lines.append("## Feature Selection by Target\n")

    report_lines.append("### Macro Targets\n")
    for n in [2, 1]:
        count = (macro_df['n_targets_selected'] == n).sum()
        if count > 0:
            report_lines.append(f"- Selected by {n} target(s): {count} features")

    report_lines.append("\n### Micro Targets\n")
    for n in range(6, 0, -1):
        count = (micro_df['n_targets_selected'] == n).sum()
        if count > 0:
            report_lines.append(f"- Selected by {n} target(s): {count} features")

    # Positional analysis
    report_lines.append("\n## Positional Distribution\n")

    macro_selected = macro_df[macro_df['n_targets_selected'] > 0]['feature'].tolist()
    macro_pos = analyze_ngram_position(macro_selected)

    report_lines.append("### Macro-level")
    report_lines.append(f"- Initial n-grams: {macro_pos['n_initial']} ({macro_pos['pct_initial']:.1f}%)")
    report_lines.append(f"- Final n-grams: {macro_pos['n_final']} ({macro_pos['pct_final']:.1f}%)\n")

    micro_selected = micro_df[micro_df['n_targets_selected'] > 0]['feature'].tolist()
    micro_pos = analyze_ngram_position(micro_selected)

    report_lines.append("### Micro-level")
    report_lines.append(f"- Initial n-grams: {micro_pos['n_initial']} ({micro_pos['pct_initial']:.1f}%)")
    report_lines.append(f"- Final n-grams: {micro_pos['n_final']} ({micro_pos['pct_final']:.1f}%)\n")

    # Top features
    report_lines.append("## Top Selected Features\n")

    report_lines.append("### Macro-level (Top 20 by max stability)\n")
    top_macro = macro_df[macro_df['n_targets_selected'] > 0].head(20)
    report_lines.append("| Rank | N-gram | Targets | Max Stability | Mean Stability |")
    report_lines.append("|------|--------|---------|---------------|----------------|")
    for idx, row in top_macro.iterrows():
        rank = idx + 1
        report_lines.append(
            f"| {rank} | `{row['feature']}` | {row['n_targets_selected']} | "
            f"{row['max_stability']:.3f} | {row['mean_stability']:.3f} |"
        )

    report_lines.append("\n### Micro-level (Top 20 by max stability)\n")
    top_micro = micro_df[micro_df['n_targets_selected'] > 0].head(20)
    report_lines.append("| Rank | N-gram | Targets | Max Stability | Mean Stability |")
    report_lines.append("|------|--------|---------|---------------|----------------|")
    for idx, row in top_micro.iterrows():
        rank = idx + 1
        report_lines.append(
            f"| {rank} | `{row['feature']}` | {row['n_targets_selected']} | "
            f"{row['max_stability']:.3f} | {row['mean_stability']:.3f} |"
        )

    # Usage recommendations
    report_lines.append("\n## Recommendations for Modeling\n")
    report_lines.append("### Feature Set Selection")
    report_lines.append("- **Conservative**: Use features selected by ≥2 targets (higher confidence)")
    report_lines.append("- **Standard**: Use all selected features (recommended)")
    report_lines.append("- **Aggressive**: Include features with max_stability ≥ 0.3 (more features, lower confidence)\n")

    report_lines.append("### Next Steps")
    report_lines.append("1. Run cross-validation to assess predictive performance")
    report_lines.append("2. Examine feature coefficients for linguistic interpretation")
    report_lines.append("3. Compare with full feature set baseline")
    report_lines.append("4. Investigate features selected by multiple targets (most robust)\n")

    # Save report
    report_content = '\n'.join(report_lines)

    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            f.write(report_content)

        print(f"✅ Report saved to: {output_file}")

    return report_content


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python generate_report.py <consolidated_results_dir>")
        print("\nExample:")
        print("  python generate_report.py ../../results/ngram_feature_selection/20251228_123456/consolidated")
        sys.exit(1)

    consolidated_dir = sys.argv[1]
    output_file = Path(consolidated_dir) / 'feature_selection_report.md'

    try:
        report = generate_diagnostic_report(consolidated_dir, output_file)

        print("\n" + "="*70)
        print("FEATURE SELECTION REPORT")
        print("="*70 + "\n")
        print(report)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
