"""
Comprehensive Results Summary: Baselines vs Experimental Models
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

RESULTS_DIR = Path(__file__).parent / 'results'

# All domains
DOMAINS = [
    'macro_has_suffix', 'macro_has_mutation', 'macro_3way',
    'micro_ablaut', 'micro_templatic', 'micro_medial_a',
    'micro_final_a', 'micro_final_vw', 'micro_insert_c', 'micro_8way'
]

# All models
BASELINES = ['majority_class', 'random', 'ngram_only']
EXPERIMENTAL = ['logistic_regression', 'random_forest', 'xgboost']
ALL_MODELS = BASELINES + EXPERIMENTAL


def get_latest_result(domain, model):
    """Get the latest result file for a domain/model combination."""
    domain_dir = RESULTS_DIR / domain
    if not domain_dir.exists():
        return None

    pattern = f"{model}_*.json"
    files = list(domain_dir.glob(pattern))
    if not files:
        return None

    # Get most recent file
    latest = max(files, key=lambda p: p.stat().st_mtime)

    with open(latest) as f:
        return json.load(f)


def collect_all_results():
    """Collect results for all models and domains."""
    results = []

    for domain in DOMAINS:
        for model in ALL_MODELS:
            result = get_latest_result(domain, model)
            if result:
                row = {
                    'domain': domain,
                    'model': model,
                    'macro_f1_mean': result['overall_metrics']['macro_f1_mean'],
                    'macro_f1_std': result['overall_metrics']['macro_f1_std'],
                    'accuracy_mean': result['overall_metrics']['accuracy_mean'],
                    'accuracy_std': result['overall_metrics']['accuracy_std'],
                }

                # Add best params for experimental models
                if 'best_params' in result:
                    row['best_params'] = str(result['best_params'])

                results.append(row)

    return pd.DataFrame(results)


def create_comparison_table(df):
    """Create comparison table: Baselines vs Full Models."""
    # Pivot table with models as columns
    pivot = df.pivot(index='domain', columns='model', values='macro_f1_mean')

    # Reorder columns
    column_order = BASELINES + EXPERIMENTAL
    pivot = pivot[[col for col in column_order if col in pivot.columns]]

    # Add delta columns (experimental vs n-gram baseline)
    if 'ngram_only' in pivot.columns:
        for exp_model in EXPERIMENTAL:
            if exp_model in pivot.columns:
                pivot[f'{exp_model}_delta'] = pivot[exp_model] - pivot['ngram_only']

    return pivot


def create_best_model_table(df):
    """Create table showing best model per domain."""
    best_models = []

    for domain in DOMAINS:
        domain_data = df[df['domain'] == domain]
        if len(domain_data) > 0:
            best_idx = domain_data['macro_f1_mean'].idxmax()
            best = domain_data.loc[best_idx]

            best_models.append({
                'domain': domain,
                'best_model': best['model'],
                'macro_f1': best['macro_f1_mean'],
                'std': best['macro_f1_std']
            })

    return pd.DataFrame(best_models)


def main():
    print("="*80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("Baselines vs Experimental Models (All Features)")
    print("="*80)

    # Collect all results
    df = collect_all_results()

    print(f"\nTotal experiments: {len(df)}")
    print(f"Domains: {len(DOMAINS)}")
    print(f"Models per domain: {len(ALL_MODELS)}")

    # Save full results
    df.to_csv(RESULTS_DIR / 'all_results_summary.csv', index=False)
    print(f"\n✓ Saved: {RESULTS_DIR / 'all_results_summary.csv'}")

    # Create comparison table
    comparison = create_comparison_table(df)
    comparison.to_csv(RESULTS_DIR / 'model_comparison.csv')
    print(f"✓ Saved: {RESULTS_DIR / 'model_comparison.csv'}")

    # Create best model table
    best_models = create_best_model_table(df)
    best_models.to_csv(RESULTS_DIR / 'best_models_per_domain.csv', index=False)
    print(f"✓ Saved: {RESULTS_DIR / 'best_models_per_domain.csv'}")

    # Print comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON (Macro-F1 Scores)")
    print("="*80)
    print("\nBaselines vs Full Models:")
    print(comparison[BASELINES + EXPERIMENTAL].round(4).to_string())

    # Print delta analysis
    if 'ngram_only' in comparison.columns:
        print("\n" + "="*80)
        print("DELTA vs N-gram Baseline")
        print("="*80)
        delta_cols = [col for col in comparison.columns if '_delta' in col]
        if delta_cols:
            deltas = comparison[delta_cols].round(4)
            deltas.columns = [col.replace('_delta', '') for col in deltas.columns]
            print(deltas.to_string())

            print("\n" + "-"*80)
            print("Summary Statistics (vs N-gram baseline):")
            for col in deltas.columns:
                mean_delta = deltas[col].mean()
                wins = (deltas[col] > 0).sum()
                losses = (deltas[col] < 0).sum()
                ties = (deltas[col] == 0).sum()
                print(f"\n{col}:")
                print(f"  Mean delta: {mean_delta:+.4f}")
                print(f"  Win/Loss/Tie: {wins}/{losses}/{ties}")

    # Print best models
    print("\n" + "="*80)
    print("BEST MODEL PER DOMAIN")
    print("="*80)
    print(best_models.to_string(index=False))

    # Model type summary
    print("\n" + "="*80)
    print("BEST MODEL TYPE FREQUENCY")
    print("="*80)
    model_counts = best_models['best_model'].value_counts()
    print(model_counts.to_string())


if __name__ == '__main__':
    main()
