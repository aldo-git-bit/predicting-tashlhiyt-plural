"""
Extract Full N-gram Vocabulary (No LASSO Selection)

Extracts ALL n-grams (1-3 phonemes) from word edges for macro-level data.
No feature selection applied - this is the pure surface-form baseline.

Output: Full n-gram feature matrix for macro-level (should be ~2,265 features)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add ngram_feature_selection to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / 'ngram_feature_selection'))

from ngram_extractor import extract_all_ngrams

# Paths
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
DATASET_PATH = DATA_DIR / 'tash_nouns.csv'
OUTPUT_PATH = DATA_DIR / 'ngram_features_macro_FULL.csv'


def main():
    print("="*80)
    print("EXTRACT FULL N-GRAM VOCABULARY (MACRO-LEVEL)")
    print("="*80)

    # Load dataset
    print(f"\nLoading dataset: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f"  Total records: {len(df)}")

    # Filter to macro-level (External, Internal, Mixed)
    macro_df = df[df['analysisPluralPattern'].isin(['External', 'Internal', 'Mixed'])].copy()
    print(f"  Macro-level records: {len(macro_df)}")

    # Extract n-grams from each stem
    print("\nExtracting n-grams...")
    ngram_lists = []
    valid_indices = []

    for idx, row in macro_df.iterrows():
        stem = row.get('analysisSingularTheme', '')

        if pd.isna(stem) or stem == '':
            continue

        ngrams = extract_all_ngrams(stem, max_n=3)
        ngram_lists.append(ngrams)
        valid_indices.append(row['recordID'])

    print(f"  Valid stems: {len(ngram_lists)}")

    # Build vocabulary (all unique n-grams)
    print("\nBuilding vocabulary...")
    vocabulary = set()
    for ngrams in ngram_lists:
        vocabulary.update(ngrams)

    vocabulary = sorted(vocabulary)
    print(f"  Total unique n-grams: {len(vocabulary)}")

    # Create feature matrix
    print("\nCreating feature matrix...")
    feature_matrix = np.zeros((len(ngram_lists), len(vocabulary)), dtype=np.int8)

    for i, ngrams in enumerate(ngram_lists):
        for ngram in ngrams:
            j = vocabulary.index(ngram)
            feature_matrix[i, j] = 1

    # Create DataFrame
    ngram_df = pd.DataFrame(
        feature_matrix,
        index=valid_indices,
        columns=vocabulary
    )

    # Save
    print(f"\nSaving: {OUTPUT_PATH}")
    ngram_df.to_csv(OUTPUT_PATH)

    # Statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"  Samples: {ngram_df.shape[0]}")
    print(f"  Features: {ngram_df.shape[1]}")
    print(f"  Mean features per sample: {feature_matrix.sum(axis=1).mean():.2f}")
    print(f"  Sparsity: {(1 - feature_matrix.mean()):.2%}")

    # Most common n-grams
    print("\nMost common n-grams:")
    ngram_counts = feature_matrix.sum(axis=0)
    top_indices = np.argsort(ngram_counts)[::-1][:10]
    for idx in top_indices:
        print(f"  {vocabulary[idx]}: {ngram_counts[idx]}")

    print("\n" + "="*80)
    print("✓ Full n-gram vocabulary extracted")
    print("="*80)


if __name__ == '__main__':
    main()
