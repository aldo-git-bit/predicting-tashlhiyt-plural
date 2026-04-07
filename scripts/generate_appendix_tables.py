#!/usr/bin/env python3
"""
Generate Appendix Tables for Model Performance.

Creates 10 tables (one per domain) showing Accuracy, Macro-F1, and AUC-ROC
for all feature sets and models.

Output: CSV files for each domain in reports/appendix_tables/
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Domain definitions
MACRO_DOMAINS = ['has_suffix', 'has_mutation', '3way']
MICRO_DOMAINS = ['medial_a', 'final_a', 'final_vw', 'ablaut', 'insert_c', 'templatic', '8way']
ALL_DOMAINS = MACRO_DOMAINS + MICRO_DOMAINS

# Domain labels and sample sizes
DOMAIN_INFO = {
    'has_suffix': {'label': 'Has Suffix', 'n': 1185, 'level': 'macro'},
    'has_mutation': {'label': 'Has Mutation', 'n': 1185, 'level': 'macro'},
    '3way': {'label': '3-way Classification', 'n': 1185, 'level': 'macro'},
    'medial_a': {'label': 'Medial A', 'n': 562, 'level': 'micro', 'smote': True, 'minority_pct': 16.5},
    'final_a': {'label': 'Final A', 'n': 562, 'level': 'micro', 'smote': True, 'minority_pct': 12.6},
    'final_vw': {'label': 'Final V/W', 'n': 562, 'level': 'micro', 'smote': True, 'minority_pct': 4.8},
    'ablaut': {'label': 'Ablaut', 'n': 562, 'level': 'micro'},
    'insert_c': {'label': 'Insert C', 'n': 562, 'level': 'micro', 'smote': True, 'minority_pct': 6.8},
    'templatic': {'label': 'Templatic', 'n': 562, 'level': 'micro'},
    '8way': {'label': '8-way Classification', 'n': 562, 'level': 'micro'}
}

# Feature set display order
FEATURE_SETS = ['semantic_only', 'morph_only', 'phon_only', 'morph_phon', 'all_features', 'ngrams_only']
FEATURE_LABELS = {
    'semantic_only': 'Semantic',
    'morph_only': 'Morphological',
    'phon_only': 'Phonological',
    'morph_phon': 'Morph+Phon',
    'all_features': 'All Features',
    'ngrams_only': 'N-grams Only'
}

# Model display order
MODELS = ['logistic_regression', 'random_forest', 'xgboost']
MODEL_LABELS = {
    'logistic_regression': 'LogReg',
    'random_forest': 'RandForest',
    'xgboost': 'XGBoost'
}


def load_ablation_results(domain):
    """Load latest ablation results for a domain."""
    results_dir = PROJECT_ROOT / 'experiments' / 'results' / f'ablation_{domain}'
    summary_files = sorted(results_dir.glob('ablation_summary_*.csv'))

    if not summary_files:
        print(f"Warning: No results found for {domain}")
        return None

    latest = summary_files[-1]
    df = pd.read_csv(latest)

    return df


def format_mean_std(mean, std, decimals=1, as_percent=True):
    """Format mean ± std as string."""
    if pd.isna(mean) or pd.isna(std):
        return "—"

    if as_percent:
        mean_pct = mean * 100
        std_pct = std * 100
        return f"{mean_pct:.{decimals}f} ± {std_pct:.{decimals}f}"
    else:
        return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def check_high_variation(std, threshold=0.15):
    """Check if standard deviation exceeds threshold (15 percentage points)."""
    if pd.isna(std):
        return False
    return std >= threshold


def generate_table_for_domain(domain):
    """Generate appendix table for a single domain."""

    # Load results
    df = load_ablation_results(domain)
    if df is None:
        return None

    # Initialize table rows
    rows = []

    for feature_set in FEATURE_SETS:
        for model in MODELS:
            # Filter to this feature set + model
            row_data = df[(df['feature_set'] == feature_set) & (df['model'] == model)]

            if len(row_data) == 0:
                continue

            row_data = row_data.iloc[0]

            # Extract metrics
            acc_mean = row_data['accuracy_mean']
            acc_std = row_data['accuracy_std']
            f1_mean = row_data['macro_f1_mean']
            f1_std = row_data['macro_f1_std']
            auc_mean = row_data.get('auc_roc_mean', np.nan)
            auc_std = row_data.get('auc_roc_std', np.nan)

            # Check for high variation
            high_var_acc = check_high_variation(acc_std)
            high_var_f1 = check_high_variation(f1_std)
            high_var_auc = check_high_variation(auc_std)

            # Format values
            acc_str = format_mean_std(acc_mean, acc_std)
            f1_str = format_mean_std(f1_mean, f1_std)
            auc_str = format_mean_std(auc_mean, auc_std)

            # Add markers for high variation
            if high_var_acc:
                acc_str += "‡"
            if high_var_f1:
                f1_str += "‡"
            if high_var_auc:
                auc_str += "‡"

            rows.append({
                'Feature Set': FEATURE_LABELS[feature_set],
                'Model': MODEL_LABELS[model],
                'Accuracy': acc_str,
                'Macro-F1': f1_str,
                'AUC-ROC': auc_str
            })

    # Create DataFrame
    table = pd.DataFrame(rows)

    return table


def generate_all_tables():
    """Generate all 10 appendix tables."""

    output_dir = PROJECT_ROOT / 'reports' / 'appendix_tables'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Appendix Tables...")
    print("="*80)

    for idx, domain in enumerate(ALL_DOMAINS, start=1):
        table_num = idx
        print(f"\nTable A{table_num}: {DOMAIN_INFO[domain]['label']} (n={DOMAIN_INFO[domain]['n']})")

        table = generate_table_for_domain(domain)

        if table is None:
            print(f"  ⚠️  Skipped (no data)")
            continue

        # Save to CSV
        output_path = output_dir / f'table_a{table_num}_{domain}.csv'
        table.to_csv(output_path, index=False)

        print(f"  ✓ Saved: {output_path.name}")
        print(f"  Rows: {len(table)}")

    print("\n" + "="*80)
    print(f"✓ All tables saved to: {output_dir}")

    # Generate metadata file
    metadata = {
        'tables': []
    }

    for idx, domain in enumerate(ALL_DOMAINS, start=1):
        info = DOMAIN_INFO[domain]
        metadata['tables'].append({
            'table_num': idx,
            'domain': domain,
            'label': info['label'],
            'n': info['n'],
            'level': info['level'],
            'smote': info.get('smote', False),
            'minority_pct': info.get('minority_pct', None)
        })

    metadata_path = output_dir / 'metadata.json'
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Metadata saved to: {metadata_path.name}")


if __name__ == '__main__':
    generate_all_tables()
