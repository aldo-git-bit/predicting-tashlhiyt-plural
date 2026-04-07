#!/usr/bin/env python3
"""
Re-run Morph+Phon Logistic Regression for all 10 domains with enhanced saving.

This script re-runs ONLY the Morph+Phon LogReg experiments to generate
results with record IDs and probabilities for residual analysis.

Usage:
    python experiments/rerun_morph_phon_logreg.py --domain has_suffix
    python experiments/rerun_morph_phon_logreg.py --all
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_ablation import run_logistic_regression, load_config, print_cv_summary
from experiments.utils import save_results
from datetime import datetime

def main():
    """Re-run Morph+Phon LogReg for specified domains."""

    parser = argparse.ArgumentParser(description='Re-run Morph+Phon LogReg with enhanced saving')
    parser.add_argument('--domain', type=str,
                        choices=['has_suffix', 'has_mutation', '3way', 'ablaut', 'medial_a',
                                'final_a', 'final_vw', 'insert_c', 'templatic', '8way'],
                        help='Domain to train')
    parser.add_argument('--all', action='store_true',
                        help='Train all 10 domains')
    parser.add_argument('--use-smote', action='store_true',
                        help='Force SMOTE usage (default: auto-determined)')

    args = parser.parse_args()

    # Determine domains
    if args.all:
        domains = ['has_suffix', 'has_mutation', '3way', 'ablaut', 'medial_a',
                  'final_a', 'final_vw', 'insert_c', 'templatic', '8way']
    elif args.domain:
        domains = [args.domain]
    else:
        parser.error("Must specify --domain or --all")

    # SMOTE configuration: Same as LSTM training
    use_smote_dict = {
        'has_suffix': args.use_smote,
        'has_mutation': args.use_smote,
        '3way': args.use_smote,
        'ablaut': args.use_smote,
        'medial_a': args.use_smote,
        'final_a': True,  # Always use SMOTE for extreme imbalance
        'final_vw': True,
        'insert_c': True,
        'templatic': args.use_smote,
        '8way': args.use_smote
    }

    # Load config
    config = load_config()

    print("="*80)
    print(f"RE-RUNNING MORPH+PHON LOGISTIC REGRESSION ({len(domains)} DOMAIN{'S' if len(domains) > 1 else ''})")
    print("="*80)
    print("\nThis will regenerate Morph+Phon LogReg results with:")
    print("  - Record IDs for each prediction")
    print("  - Full probability distributions")
    print("  - Required for residual analysis")
    print()

    start_time = datetime.now()

    for domain in domains:
        print(f"\n{'#'*80}")
        print(f"# DOMAIN: {domain.upper()}")
        print(f"{'#'*80}")

        # Run Morph+Phon LogReg
        use_smote = use_smote_dict.get(domain, False)
        if use_smote:
            print(f"  Using SMOTE for {domain} (extreme class imbalance)")

        results = run_logistic_regression(
            domain=domain,
            feature_set='morph_phon',
            config=config,
            use_smote=use_smote
        )

        # Save results with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_results(results, f'ablation_{domain}',
                    'logistic_regression_morph_phon',
                    timestamp)

        print(f"\n✓ Completed {domain}")
        print(f"  - Record IDs saved: {len(results['predictions']['record_ids'])}")
        print(f"  - Probabilities saved: {len(results['predictions']['y_proba'])}")
        print(f"  - Mean Macro-F1: {results['overall_metrics']['macro_f1_mean']:.3f}")

    elapsed = (datetime.now() - start_time).total_seconds() / 60
    print("\n" + "="*80)
    print(f"✅ ALL {len(domains)} DOMAIN{'S' if len(domains) > 1 else ''} COMPLETE ({elapsed:.1f} minutes)")
    print("="*80)
    print("\nResults saved with enhanced data for residual analysis:")
    for domain in domains:
        results_dir = PROJECT_ROOT / 'experiments' / 'results' / f'ablation_{domain}'
        latest_file = max(results_dir.glob('logistic_regression_morph_phon_*.json'))
        print(f"  - {domain}: {latest_file.name}")

if __name__ == '__main__':
    main()
