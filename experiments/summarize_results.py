"""
Generate summary tables of experiment results.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List

from utils import RESULTS_DIR, load_config


def collect_baseline_results() -> pd.DataFrame:
    """
    Collect all baseline results into a single DataFrame.

    Returns:
        DataFrame with all baseline results
    """
    config = load_config()
    all_domains = []
    for level in ['macro', 'micro']:
        for domain in config['domains'][level]:
            all_domains.append(f"{level}_{domain}")

    baselines = ['majority_class', 'random', 'ngram_only']

    results = []

    for domain in all_domains:
        domain_dir = RESULTS_DIR / domain

        if not domain_dir.exists():
            print(f"⚠️  No results for {domain}")
            continue

        for baseline in baselines:
            # Find most recent file
            files = list(domain_dir.glob(f"{baseline}_*.json"))

            if not files:
                print(f"⚠️  No {baseline} results for {domain}")
                continue

            latest_file = max(files, key=lambda p: p.stat().st_mtime)

            # Load results
            with open(latest_file, 'r') as f:
                data = json.load(f)

            # Extract key metrics
            metrics = data['overall_metrics']

            row = {
                'domain': domain,
                'level': domain.split('_')[0],
                'task': '_'.join(domain.split('_')[1:]),
                'model': baseline,
                'macro_f1_mean': metrics['macro_f1_mean'],
                'macro_f1_std': metrics['macro_f1_std'],
                'accuracy_mean': metrics['accuracy_mean'],
                'accuracy_std': metrics['accuracy_std'],
                'n_samples': data['n_samples'],
                'n_features': data['n_features']
            }

            results.append(row)

    df = pd.DataFrame(results)
    return df


def print_baseline_summary(df: pd.DataFrame):
    """
    Print formatted summary of baseline results.

    Args:
        df: DataFrame with baseline results
    """
    print("\n" + "="*100)
    print("BASELINE RESULTS SUMMARY")
    print("="*100)

    for level in ['macro', 'micro']:
        level_df = df[df['level'] == level]

        if len(level_df) == 0:
            continue

        print(f"\n{level.upper()}-LEVEL TASKS:")
        print("-"*100)

        for task in sorted(level_df['task'].unique()):
            task_df = level_df[level_df['task'] == task]

            print(f"\n{task}:")
            print(f"  {'Model':<25} {'Macro-F1':<20} {'Accuracy':<20} {'N':<10}")
            print("  " + "-"*85)

            for _, row in task_df.iterrows():
                model_name = row['model'].replace('_', ' ').title()
                f1_str = f"{row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f}"
                acc_str = f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}"

                print(f"  {model_name:<25} {f1_str:<20} {acc_str:<20} {int(row['n_samples']):<10}")

    print("\n" + "="*100)


def generate_baseline_comparison_table(df: pd.DataFrame, output_format='markdown') -> str:
    """
    Generate comparison table of baseline results.

    Args:
        df: DataFrame with baseline results
        output_format: 'markdown' or 'latex'

    Returns:
        Formatted table string
    """
    # Pivot to show baselines as columns
    pivot = df.pivot_table(
        index=['level', 'task'],
        columns='model',
        values='macro_f1_mean',
        aggfunc='first'
    )

    # Reorder columns
    col_order = ['majority_class', 'random', 'ngram_only']
    pivot = pivot[[col for col in col_order if col in pivot.columns]]

    # Rename columns
    pivot.columns = [col.replace('_', ' ').title() for col in pivot.columns]

    # Format as markdown
    if output_format == 'markdown':
        table = pivot.to_markdown(floatfmt=".4f")
    elif output_format == 'latex':
        table = pivot.to_latex(float_format="%.4f")
    else:
        table = str(pivot)

    return table


def save_baseline_summary(df: pd.DataFrame, output_dir: Path = None):
    """
    Save baseline summary to CSV and markdown.

    Args:
        df: DataFrame with baseline results
        output_dir: Output directory (default: experiments/results)
    """
    if output_dir is None:
        output_dir = RESULTS_DIR

    # Save full results to CSV
    csv_path = output_dir / 'baseline_results_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved CSV: {csv_path}")

    # Save markdown table
    md_table = generate_baseline_comparison_table(df, output_format='markdown')
    md_path = output_dir / 'baseline_results_comparison.md'
    with open(md_path, 'w') as f:
        f.write("# Baseline Results Comparison\n\n")
        f.write("**Metric**: Macro-F1 (mean across 10-fold CV)\n\n")
        f.write(md_table)
    print(f"✓ Saved Markdown: {md_path}")

    # Save LaTeX table
    latex_table = generate_baseline_comparison_table(df, output_format='latex')
    tex_path = output_dir / 'baseline_results_comparison.tex'
    with open(tex_path, 'w') as f:
        f.write(latex_table)
    print(f"✓ Saved LaTeX: {tex_path}")


if __name__ == '__main__':
    print("Collecting baseline results...")
    df = collect_baseline_results()

    print(f"\nCollected results for {len(df)} experiments")

    # Print summary
    print_baseline_summary(df)

    # Save summary
    save_baseline_summary(df)

    print("\n✓ Baseline summary complete")
