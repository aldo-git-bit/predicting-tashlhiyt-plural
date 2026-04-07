"""
Quick Pipeline Test

Tests the complete feature selection pipeline with reduced iterations.
"""

import pandas as pd
from run_all_targets import run_all_targets

if __name__ == '__main__':
    print("\n" + "="*70)
    print("PIPELINE TEST (10 iterations per target)")
    print("="*70 + "\n")

    # Load dataset
    df = pd.read_csv('../../data/tash_nouns.csv')

    # Run with only 10 iterations for testing
    results = run_all_targets(
        df,
        n_iterations=10,
        output_dir='../../results/ngram_feature_selection_test'
    )

    print("\n✅ Pipeline test complete")
