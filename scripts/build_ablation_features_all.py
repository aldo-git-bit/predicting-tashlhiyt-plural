"""
Build Feature Matrices for Ablation Study - All Domains

Generates 6 feature sets for any domain:
1. N-grams only (baseline): Full n-grams for the level (macro/micro)
2. Semantic only: 24 features
3. Morphological only: 12 features
4. Phonological only: 9 prosodic + task-specific n-grams
5. Morph + Phon: 12 + 9 + task-specific n-grams
6. All features: 12 + 24 + 9 + task-specific n-grams

Usage:
    python scripts/build_ablation_features_all.py --domain has_suffix
    python scripts/build_ablation_features_all.py --domain medial_a
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
FEATURES_DIR = PROJECT_ROOT / 'features'
NGRAM_SELECTION_DIR = PROJECT_ROOT / 'results' / 'ngram_feature_selection' / '20251228_204410'

# Domain configuration
DOMAIN_CONFIG = {
    # Macro domains
    'has_suffix': {
        'level': 'macro',
        'target_file': 'y_macro_has_suffix',
        'ngram_selection': 'macro/y_macro_suffix_selected.txt',
        'full_ngrams': 'ngram_features_macro_FULL.csv'
    },
    'has_mutation': {
        'level': 'macro',
        'target_file': 'y_macro_has_mutation',
        'ngram_selection': 'macro/y_macro_mutated_selected.txt',
        'full_ngrams': 'ngram_features_macro_FULL.csv'
    },
    '3way': {
        'level': 'macro',
        'target_file': 'y_macro_3way',
        'ngram_selection': 'consolidated/macro_final_features.txt',
        'full_ngrams': 'ngram_features_macro_FULL.csv'
    },
    # Micro domains
    'medial_a': {
        'level': 'micro',
        'target_file': 'y_micro_medial_a',
        'ngram_selection': 'micro/y_micro_medial_a_selected.txt',
        'full_ngrams': 'ngram_features_micro.csv'
    },
    'final_a': {
        'level': 'micro',
        'target_file': 'y_micro_final_a',
        'ngram_selection': 'micro/y_micro_final_a_selected.txt',
        'full_ngrams': 'ngram_features_micro.csv'
    },
    'final_vw': {
        'level': 'micro',
        'target_file': 'y_micro_final_vw',
        'ngram_selection': 'micro/y_micro_final_vw_selected.txt',
        'full_ngrams': 'ngram_features_micro.csv'
    },
    'ablaut': {
        'level': 'micro',
        'target_file': 'y_micro_ablaut',
        'ngram_selection': 'micro/y_micro_ablaut_selected.txt',
        'full_ngrams': 'ngram_features_micro.csv'
    },
    'insert_c': {
        'level': 'micro',
        'target_file': 'y_micro_insert_c',
        'ngram_selection': 'micro/y_micro_insert_c_selected.txt',
        'full_ngrams': 'ngram_features_micro.csv'
    },
    'templatic': {
        'level': 'micro',
        'target_file': 'y_micro_templatic',
        'ngram_selection': 'micro/y_micro_templatic_selected.txt',
        'full_ngrams': 'ngram_features_micro.csv'
    },
    '8way': {
        'level': 'micro',
        'target_file': 'y_micro_8way',
        'ngram_selection': 'consolidated/micro_final_features.txt',
        'full_ngrams': 'ngram_features_micro.csv'
    }
}


def build_ablation_features(domain: str):
    """Build ablation feature sets for a specific domain."""

    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(DOMAIN_CONFIG.keys())}")

    config = DOMAIN_CONFIG[domain]
    level = config['level']

    print("="*80)
    print(f"BUILD ABLATION FEATURE MATRICES: {domain}")
    print("="*80)
    print(f"  Level: {level}")
    print(f"  Target: {config['target_file']}")
    print()

    # Create output directory
    output_dir = FEATURES_DIR / f'ablation_{domain}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing feature matrix for this domain
    print(f"Loading existing features: X_{level}_{domain.replace('-', '_')}.csv")
    X_existing_path = FEATURES_DIR / f"X_{level}_{domain.replace('-', '_')}.csv"
    X_existing = pd.read_csv(X_existing_path, index_col=0)
    print(f"  Shape: {X_existing.shape}")

    # Load target
    print(f"\nLoading target: {config['target_file']}.csv")
    y_path = FEATURES_DIR / f"{config['target_file']}.csv"
    y = pd.read_csv(y_path, index_col=0).squeeze()
    print(f"  Target shape: {y.shape}")

    # Identify existing features by prefix
    m_cols = [col for col in X_existing.columns if col.startswith('m_')]
    s_cols = [col for col in X_existing.columns if col.startswith('s_')]
    p_cols_old = [col for col in X_existing.columns if col.startswith('p_')]

    print(f"\nExisting features:")
    print(f"  Morphological: {len(m_cols)}")
    print(f"  Semantic: {len(s_cols)}")
    print(f"  Phonological (old): {len(p_cols_old)}")

    # Load main dataset to get NEW prosodic features
    print(f"\nLoading main dataset for new prosodic features")
    df = pd.read_csv(DATA_DIR / 'tash_nouns.csv')
    df = df.set_index('recordID')

    # Get new prosodic features
    new_prosodic = ['p_LH_count_heavies', 'p_LH_count_moras', 'p_foot_count_feet', 'p_foot_residue']
    prosodic_new = df.loc[X_existing.index, new_prosodic]
    print(f"  New prosodic features: {len(new_prosodic)}")

    # Map old prosodic names to new names
    prosodic_rename = {
        'p_lh_final_weight': 'p_LH_ends_L',
        'p_lh_initial_weight': 'p_LH_initial_weight',
        'p_lh_less_2_syllables': 'p_LH_less_2_syllables',
        'p_lh_all_heavy': 'p_LH_all_heavy',
        'p_foot_unparsed_final': 'p_foot_residue_right'
    }

    # Extract and rename old prosodic
    prosodic_old_renamed = X_existing[p_cols_old].rename(columns=prosodic_rename)

    # Remove p_lh_all_light if present (not in the plan)
    if 'p_lh_all_light' in prosodic_old_renamed.columns:
        prosodic_old_renamed = prosodic_old_renamed.drop(columns=['p_lh_all_light'])

    # FIX BUG 1: Flip p_LH_ends_L values
    # Original p_lh_final_weight: 1=ends in H, 0=ends in L
    # New p_LH_ends_L should be: 1=ends in L, 0=ends in H
    if 'p_LH_ends_L' in prosodic_old_renamed.columns:
        prosodic_old_renamed['p_LH_ends_L'] = 1 - prosodic_old_renamed['p_LH_ends_L']
        print(f"  ✓ Fixed p_LH_ends_L: flipped values (1=ends in L, 0=ends in H)")

    # FIX BUG 2: Regenerate p_foot_residue_right from main dataset
    # Original feature was corrupted; regenerate correctly
    if 'p_foot_residue_right' in prosodic_old_renamed.columns:
        prosodic_old_renamed['p_foot_residue_right'] = df.loc[X_existing.index, 'p_stem_sing_foot'].str.endswith('l').fillna(False).astype(int)
        print(f"  ✓ Fixed p_foot_residue_right: regenerated from p_stem_sing_foot")

    # Combine old + new prosodic
    prosodic_all = pd.concat([prosodic_old_renamed, prosodic_new], axis=1)
    print(f"  Combined prosodic features: {prosodic_all.shape[1]}")

    # Load full n-grams for this level
    print(f"\nLoading full n-grams: {config['full_ngrams']}")
    ngrams_full = pd.read_csv(DATA_DIR / config['full_ngrams'], index_col=0)
    ngrams_full = ngrams_full.loc[X_existing.index]  # Align indices
    print(f"  Shape: {ngrams_full.shape}")

    # Load task-specific n-grams
    ngram_selection_path = NGRAM_SELECTION_DIR / config['ngram_selection']
    print(f"\nLoading task-specific n-grams: {ngram_selection_path.name}")
    with open(ngram_selection_path) as f:
        selected_ngrams = [line.strip() for line in f if line.strip()]
    print(f"  Selected n-grams: {len(selected_ngrams)}")

    # Filter full n-grams to task-specific
    ngrams_selected = ngrams_full[selected_ngrams]
    print(f"  Task-specific matrix: {ngrams_selected.shape}")

    # Save target to ablation directory
    y.to_csv(output_dir / f'y_{domain}.csv', header=[config['target_file'].split('_')[-1]])

    # Build 6 feature matrices
    print("\n" + "="*80)
    print("BUILDING FEATURE MATRICES")
    print("="*80)

    # 1. N-grams only (baseline): Full n-grams for level
    print("\n1. N-grams only (baseline)")
    X1 = ngrams_full.copy()
    print(f"   Shape: {X1.shape}")
    X1.to_csv(output_dir / 'X_ngrams_only.csv')

    # 2. Semantic only: 24 features
    print("\n2. Semantic only")
    X2 = X_existing[s_cols].copy()
    print(f"   Shape: {X2.shape}")
    X2.to_csv(output_dir / 'X_semantic_only.csv')

    # 3. Morphological only: 12 features
    print("\n3. Morphological only")
    X3 = X_existing[m_cols].copy()
    print(f"   Shape: {X3.shape}")
    X3.to_csv(output_dir / 'X_morph_only.csv')

    # 4. Phonological only: 9 prosodic + task-specific n-grams
    print("\n4. Phonological only")
    X4 = pd.concat([prosodic_all, ngrams_selected], axis=1)
    print(f"   Shape: {X4.shape} ({prosodic_all.shape[1]} prosodic + {len(selected_ngrams)} n-grams)")
    X4.to_csv(output_dir / 'X_phon_only.csv')

    # 5. Morph + Phon: 12 + 9 + task-specific n-grams
    print("\n5. Morphological + Phonological")
    X5 = pd.concat([X_existing[m_cols], prosodic_all, ngrams_selected], axis=1)
    print(f"   Shape: {X5.shape}")
    X5.to_csv(output_dir / 'X_morph_phon.csv')

    # 6. All features: 12 + 24 + 9 + task-specific n-grams
    print("\n6. All features")
    X6 = pd.concat([X_existing[m_cols], X_existing[s_cols], prosodic_all, ngrams_selected], axis=1)
    print(f"   Shape: {X6.shape}")
    X6.to_csv(output_dir / 'X_all_features.csv')

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nFeature matrices saved to: {output_dir}")
    print("\nFeature set sizes:")
    print(f"  1. N-grams only:     {X1.shape[1]:4d} features")
    print(f"  2. Semantic only:    {X2.shape[1]:4d} features")
    print(f"  3. Morph only:       {X3.shape[1]:4d} features")
    print(f"  4. Phon only:        {X4.shape[1]:4d} features ({prosodic_all.shape[1]} prosodic + {len(selected_ngrams)} n-grams)")
    print(f"  5. Morph + Phon:     {X5.shape[1]:4d} features")
    print(f"  6. All features:     {X6.shape[1]:4d} features")

    print("\nTarget variable:")
    print(f"  {config['target_file']}: {len(y)} samples")

    # Print class distribution
    if y.dtype == 'object' or len(y.unique()) <= 10:
        print(f"\nClass distribution:")
        for label, count in y.value_counts().sort_index().items():
            print(f"    {label}: {count} ({count/len(y):.1%})")

    print("\n" + "="*80)
    print("✓ Feature matrices created")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Build ablation feature matrices for any domain')
    parser.add_argument('--domain', type=str, required=True,
                       help=f'Domain name: {", ".join(DOMAIN_CONFIG.keys())}')

    args = parser.parse_args()
    build_ablation_features(args.domain)


if __name__ == '__main__':
    main()
