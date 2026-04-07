"""
Run all baseline experiments for all 10 domains.
"""

from baseline_models import run_baselines_for_all_domains
from utils import load_config

if __name__ == '__main__':
    # Load configuration
    config = load_config()

    # Run all baselines for all domains
    all_results = run_baselines_for_all_domains(config, save=True)

    print("\n" + "="*80)
    print("ALL BASELINE EXPERIMENTS COMPLETE!")
    print("="*80)
