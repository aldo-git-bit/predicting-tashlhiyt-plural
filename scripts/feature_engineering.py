#!/usr/bin/env python3
"""
Feature Engineering Pipeline for Tashlhiyt Plural Prediction

This script creates all feature matrices and target variables for the 10 model domains:
- 3 macro-level models (Has_Suffix, Has_Mutation, 3-way)
- 7 micro-level models (Medial A, Final A, Final Vw, Ablaut, Insert C, Templatic, 8-way)

Feature families:
1. Morphological (4 features → 17 dimensions after one-hot encoding + grouping)
2. Semantic (3 features → 26 dimensions after one-hot encoding)
3. Phonological (6 theory-driven binary features)
4. N-grams (model-specific: 1,004-2,019 features)

Output:
- features/X_<domain>.csv - Feature matrix for each domain
- features/y_<domain>.csv - Target variable for each domain
- features/feature_metadata_<domain>.json - Feature names and statistics
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
FEATURES_DIR = Path(__file__).resolve().parent.parent / 'features'
FEATURES_DIR.mkdir(exist_ok=True)

# Random seed for reproducibility
RANDOM_STATE = 42

# ============================================================================
# Feature Engineering Functions
# ============================================================================

def create_phonological_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 6 theory-driven phonological binary features from LH patterns.

    Features:
    1. p_lh_final_weight: Stem ends in L (0) vs H (1)
    2. p_lh_initial_weight: Stem starts with L (0) vs H (1)
    3. p_lh_all_light: All syllables are Light (1) vs mixed/heavy (0)
    4. p_lh_all_heavy: All syllables are Heavy (1) vs mixed/light (0)
    5. p_lh_less_2_syllables: Stem has ≤2 syllables (1) vs 3+ (0)
    6. p_foot_unparsed_final: Foot structure ends in 'l' (1) vs 'F' (0)

    Args:
        df: DataFrame with p_stem_sing_LH and p_stem_sing_foot columns

    Returns:
        DataFrame with 6 new phonological feature columns
    """
    print("Creating theory-driven phonological features...")

    df_phon = df.copy()

    # Feature 1: Final weight (ends in L=0, H=1)
    df_phon['p_lh_final_weight'] = df_phon['p_stem_sing_LH'].str.endswith('H').fillna(False).astype(int)

    # Feature 2: Initial weight (starts with L=0, H=1)
    df_phon['p_lh_initial_weight'] = df_phon['p_stem_sing_LH'].str.startswith('H').fillna(False).astype(int)

    # Feature 3: All light syllables
    df_phon['p_lh_all_light'] = (df_phon['p_stem_sing_LH'].str.replace('L', '').str.len() == 0).fillna(False).astype(int)

    # Feature 4: All heavy syllables
    df_phon['p_lh_all_heavy'] = (df_phon['p_stem_sing_LH'].str.replace('H', '').str.len() == 0).fillna(False).astype(int)

    # Feature 5: Stem length (≤2 syllables = 1, 3+ = 0)
    df_phon['p_lh_less_2_syllables'] = (df_phon['p_stem_sing_LH'].str.len() <= 2).fillna(False).astype(int)

    # Feature 6: Unparsed final syllable (foot ends in 'l' = 1, 'F' = 0)
    df_phon['p_foot_unparsed_final'] = df_phon['p_stem_sing_foot'].str.endswith('l').fillna(False).astype(int)

    print(f"  Created 6 phonological features")
    print(f"  Sample values:")
    for col in ['p_lh_final_weight', 'p_lh_initial_weight', 'p_lh_all_light',
                'p_lh_all_heavy', 'p_lh_less_2_syllables', 'p_foot_unparsed_final']:
        print(f"    {col}: {df_phon[col].value_counts().to_dict()}")

    return df_phon


def group_morphological_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply documented groupings to morphological features.

    Groupings:
    1. analysisMutability: Keep as-is (5 categories)
       - Drop analysisGender (redundant with mutability)
    2. wordDerivedCategory: Group 14 → 5 categories
       - Underived (1,260)
       - Action Nouns (342)
       - Agentives (177)
       - Instrumental (54) = Instrumental + Occupational
       - Tirrugza (31)
       - Other (50) = 8 rare categories merged
    3. analysisRAugVowel: Group 4 → 3 categories
       - A (merge A+I since 90% are A)
       - U
       - Zero
    4. lexiconLoanwordSource: Group 6 → 4 categories
       - Native
       - Arabic-Assimilated
       - French (merge French + Spanish)
       - Unknown (merge Unknown → Arabic-Assimilated)

    Args:
        df: DataFrame with morphological columns

    Returns:
        DataFrame with grouped morphological columns
    """
    print("Grouping morphological features...")

    df_morph = df.copy()

    # 1. Mutability: Keep as-is (already clean)
    df_morph['m_mutability'] = df_morph['analysisMutability']

    # 2. Derived Category: Group 14 → 5
    derivation_mapping = {
        'Underived': 'Underived',
        'Action Noun': 'Action Nouns',
        'Agentive': 'Agentives',
        'Instrumental': 'Instrumental',
        'Occupational': 'Instrumental',  # Merge
        'Tirrugza': 'Tirrugza',
        # Merge 8 rare categories → Other
        'Diminutive': 'Other',
        'Locative': 'Other',
        'Patient': 'Other',
        'Quality': 'Other',
        'Reciprocal': 'Other',
        'Relational': 'Other',
        'Resultative': 'Other',
        'Unit': 'Other'
    }
    df_morph['m_derivational_category'] = df_morph['wordDerivedCategory'].map(derivation_mapping).fillna('Underived')

    # 3. R-Augment Vowel: Group 4 → 3 (merge A+I → A)
    r_aug_mapping = {
        'A': 'A',
        'I': 'A',  # Merge (90% are A)
        'U': 'U',
        'Zero': 'Zero'
    }
    df_morph['m_r_aug'] = df_morph['analysisRAugVowel'].map(r_aug_mapping).fillna('Zero')

    # 4. Loanword Source: Group 6 → 4
    loan_mapping = {
        'Native': 'Native',
        'Arabic-Assimilated': 'Arabic-Assimilated',
        'French': 'French',
        'Spanish': 'French',  # Merge
        'Unknown': 'Arabic-Assimilated',  # Merge
        'Berber': 'Native'  # If exists, merge to Native
    }
    df_morph['m_loanTypes'] = df_morph['lexiconLoanwordSource'].map(loan_mapping).fillna('Native')

    print(f"  m_mutability: {df_morph['m_mutability'].nunique()} categories")
    print(f"  m_derivational_category: {df_morph['m_derivational_category'].nunique()} categories (was 14)")
    print(f"  m_r_aug: {df_morph['m_r_aug'].nunique()} categories (was 4)")
    print(f"  m_loanTypes: {df_morph['m_loanTypes'].nunique()} categories")

    return df_morph


def create_semantic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create semantic features.

    Features:
    1. s_animacy: Binary (lexiconAnimate Y/N)
    2. s_humanness: Binary (lexiconHuman Y/N)
    3. s_semantic_field: Categorical (21 categories, keep as-is)

    Note: lexiconSexGender excluded (92% unspecified)

    Args:
        df: DataFrame with semantic columns

    Returns:
        DataFrame with semantic feature columns
    """
    print("Creating semantic features...")

    df_sem = df.copy()

    # Binary features: Convert Y/N to 1/0
    df_sem['s_animacy'] = (df_sem['lexiconAnimateYN'] == 'Y').astype(int)
    df_sem['s_humanness'] = (df_sem['lexiconHumanYN'] == 'Y').astype(int)

    # Categorical feature: Keep as-is
    df_sem['s_semantic_field'] = df_sem['lexiconSemanticField']

    print(f"  s_animacy: {df_sem['s_animacy'].value_counts().to_dict()}")
    print(f"  s_humanness: {df_sem['s_humanness'].value_counts().to_dict()}")
    print(f"  s_semantic_field: {df_sem['s_semantic_field'].nunique()} categories")

    return df_sem


def create_target_variables(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Create all 10 target variables (y).

    Macro-level (n=1,185):
    1. y_macro_has_suffix: Binary (has external suffix)
    2. y_macro_has_mutation: Binary (has internal changes)
    3. y_macro_3way: 3-class (External, Internal, Mixed)

    Micro-level (n=562, only Internal/Mixed):
    4. y_micro_medial_a: Binary (has Medial A)
    5. y_micro_final_a: Binary (has Final A)
    6. y_micro_final_vw: Binary (has Final Vw)
    7. y_micro_ablaut: Binary (has Ablaut)
    8. y_micro_insert_c: Binary (has Insert C)
    9. y_micro_templatic: Binary (has Templatic)
    10. y_micro_8way: 8-class (combinations of mutations)

    Args:
        df: DataFrame with analysisPluralPattern and analysisInternalChanges

    Returns:
        Dictionary mapping target variable names to Series
    """
    print("Creating target variables...")

    targets = {}

    # Filter for usable records (exclude No Plural, Only Plural, id Plural)
    macro_mask = df['analysisPluralPattern'].isin(['External', 'Internal', 'Mixed'])
    micro_mask = df['analysisPluralPattern'].isin(['Internal', 'Mixed'])

    print(f"  Macro-level: n={macro_mask.sum()} usable records")
    print(f"  Micro-level: n={micro_mask.sum()} usable records")

    # Macro-level targets
    # 1. Has_Suffix: External or Mixed
    targets['y_macro_has_suffix'] = df.loc[macro_mask, 'analysisPluralPattern'].isin(['External', 'Mixed']).astype(int)

    # 2. Has_Mutation: Internal or Mixed
    targets['y_macro_has_mutation'] = df.loc[macro_mask, 'analysisPluralPattern'].isin(['Internal', 'Mixed']).astype(int)

    # 3. 3-way: External, Internal, Mixed
    targets['y_macro_3way'] = df.loc[macro_mask, 'analysisPluralPattern'].copy()

    # Micro-level targets (only for Internal/Mixed)
    micro_df = df[micro_mask].copy()

    # Parse analysisInternalChanges (multi-value field, newline-separated)
    def has_mutation_type(row, mutation_type):
        """Check if row has specific mutation type."""
        if pd.isna(row['analysisInternalChanges']):
            return 0
        mutations = str(row['analysisInternalChanges']).split('\n')
        return int(mutation_type in mutations)

    # 4-9. Binary mutation features
    for mutation_type in ['Medial A', 'Final A', 'Final Vw', 'Ablaut', 'Insert C', 'Templatic']:
        col_name = f"y_micro_{mutation_type.lower().replace(' ', '_')}"
        targets[col_name] = micro_df.apply(lambda row: has_mutation_type(row, mutation_type), axis=1)

    # 10. 8-way: Create multi-class from mutation combinations
    # Group by unique mutation combinations
    def get_mutation_label(row):
        """Get mutation label from combinations."""
        if pd.isna(row['analysisInternalChanges']):
            return 'None'
        mutations = sorted(str(row['analysisInternalChanges']).split('\n'))
        return ' + '.join(mutations) if mutations else 'None'

    targets['y_micro_8way'] = micro_df.apply(get_mutation_label, axis=1)

    # Print distribution summary
    print("\n  Target variable distributions:")
    for name, target in targets.items():
        if target.dtype == 'int64':
            # Binary
            dist = target.value_counts()
            print(f"    {name}: n={len(target)}, {dist.to_dict()}")
        else:
            # Multi-class
            dist = target.value_counts()
            print(f"    {name}: n={len(target)}, {dist.nunique()} classes")
            print(f"      Top 3: {dist.head(3).to_dict()}")

    return targets


def one_hot_encode_features(df: pd.DataFrame, categorical_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    One-hot encode categorical features.

    Args:
        df: DataFrame with categorical columns
        categorical_cols: List of column names to encode

    Returns:
        Tuple of (encoded DataFrame, list of new column names)
    """
    print(f"One-hot encoding {len(categorical_cols)} categorical features...")

    # Use pandas get_dummies with drop_first=False to keep all categories
    df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix_sep='__', drop_first=False)

    # Get list of new column names
    new_cols = [col for col in df_encoded.columns if '__' in col]

    print(f"  Created {len(new_cols)} one-hot encoded columns")

    return df_encoded, new_cols


def load_ngram_features(level: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load n-gram features for specified level.

    Args:
        level: 'macro' or 'micro'

    Returns:
        Tuple of (n-gram DataFrame, list of feature names)
    """
    print(f"Loading {level}-level n-gram features...")

    ngram_path = DATA_DIR / f'ngram_features_{level}.csv'
    metadata_path = DATA_DIR / f'ngram_metadata_{level}.json'

    # Load n-grams
    ngrams_df = pd.read_csv(ngram_path, index_col=0)

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    feature_names = metadata['feature_names']

    print(f"  Loaded {len(feature_names)} n-gram features")
    print(f"  Shape: {ngrams_df.shape}")

    return ngrams_df, feature_names


def create_feature_matrix(df: pd.DataFrame, domain: str, ngrams_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Create complete feature matrix for a model domain.

    Args:
        df: DataFrame with all engineered features
        domain: Model domain name (e.g., 'macro_has_suffix')
        ngrams_df: Optional n-gram DataFrame

    Returns:
        Feature matrix (X) for the domain
    """
    print(f"\nCreating feature matrix for {domain}...")

    # Select feature columns based on domain
    feature_cols = []

    # 1. Morphological features (one-hot encoded)
    morph_cols = [col for col in df.columns if col.startswith('m_') and '__' in col]
    feature_cols.extend(morph_cols)
    print(f"  Morphological: {len(morph_cols)} dimensions")

    # 2. Semantic features
    # Binary features
    sem_binary = ['s_animacy', 's_humanness']
    feature_cols.extend(sem_binary)

    # Semantic field (one-hot encoded)
    sem_field_cols = [col for col in df.columns if col.startswith('s_semantic_field__')]
    feature_cols.extend(sem_field_cols)
    print(f"  Semantic: {len(sem_binary) + len(sem_field_cols)} dimensions")

    # 3. Phonological features (6 binary features)
    phon_cols = ['p_lh_final_weight', 'p_lh_initial_weight', 'p_lh_all_light',
                 'p_lh_all_heavy', 'p_lh_less_2_syllables', 'p_foot_unparsed_final']
    feature_cols.extend(phon_cols)
    print(f"  Phonological: {len(phon_cols)} dimensions")

    # Create base feature matrix
    X = df[feature_cols].copy()

    # 4. Add n-grams if provided
    if ngrams_df is not None:
        # Align indices
        X = X.join(ngrams_df, how='inner')
        print(f"  N-grams: {ngrams_df.shape[1]} features")

    print(f"  Total features: {X.shape[1]}")
    print(f"  Total samples: {X.shape[0]}")

    return X


def save_features_and_targets(X: pd.DataFrame, y: pd.Series, domain: str):
    """
    Save feature matrix and target variable for a domain.

    Args:
        X: Feature matrix
        y: Target variable
        domain: Model domain name
    """
    # Save features
    X_path = FEATURES_DIR / f'X_{domain}.csv'
    X.to_csv(X_path, index=True)
    print(f"  Saved features: {X_path}")

    # Save target
    y_path = FEATURES_DIR / f'y_{domain}.csv'
    y.to_csv(y_path, index=True, header=True)
    print(f"  Saved target: {y_path}")

    # Save metadata
    metadata = {
        'domain': domain,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'feature_names': X.columns.tolist(),
        'target_name': y.name,
        'target_type': 'binary' if y.dtype == 'int64' else 'multiclass',
        'created_date': datetime.now().isoformat(),
        'statistics': {
            'feature_types': {
                'morphological': len([c for c in X.columns if c.startswith('m_')]),
                'semantic': len([c for c in X.columns if c.startswith('s_')]),
                'phonological': len([c for c in X.columns if c.startswith('p_')]),
                'ngrams': len([c for c in X.columns if not any(c.startswith(p) for p in ['m_', 's_', 'p_'])])
            }
        }
    }

    if y.dtype != 'int64':
        # Multi-class: add class distribution
        metadata['class_distribution'] = y.value_counts().to_dict()
    else:
        # Binary: add class counts
        metadata['class_distribution'] = {0: int((y == 0).sum()), 1: int((y == 1).sum())}

    metadata_path = FEATURES_DIR / f'feature_metadata_{domain}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main feature engineering pipeline."""

    print("="*80)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*80)

    # 1. Load main dataset
    print("\n[1/6] Loading main dataset...")
    df = pd.read_csv(DATA_DIR / 'tash_nouns.csv')

    # Set recordID as index to match n-gram files
    df = df.set_index('recordID')
    print(f"  Loaded {len(df)} records (indexed by recordID)")

    # 2. Create phonological features
    print("\n[2/6] Engineering phonological features...")
    df = create_phonological_features(df)

    # 3. Group morphological features
    print("\n[3/6] Grouping morphological features...")
    df = group_morphological_features(df)

    # 4. Create semantic features
    print("\n[4/6] Creating semantic features...")
    df = create_semantic_features(df)

    # 5. One-hot encode categorical features
    print("\n[5/6] One-hot encoding categorical features...")
    categorical_cols = ['m_mutability', 'm_derivational_category', 'm_r_aug', 'm_loanTypes', 's_semantic_field']
    df, encoded_cols = one_hot_encode_features(df, categorical_cols)

    # 6. Create target variables
    print("\n[6/6] Creating target variables...")
    targets = create_target_variables(df)

    # Load n-gram features
    print("\n" + "="*80)
    print("LOADING N-GRAM FEATURES")
    print("="*80)
    ngrams_macro, _ = load_ngram_features('macro')
    ngrams_micro, _ = load_ngram_features('micro')

    # Create feature matrices for all 10 domains
    print("\n" + "="*80)
    print("CREATING FEATURE MATRICES FOR 10 DOMAINS")
    print("="*80)

    # Get indices for macro and micro levels
    macro_idx = targets['y_macro_3way'].index
    micro_idx = targets['y_micro_ablaut'].index

    # Align with n-gram indices (only keep records that have n-gram features)
    macro_idx = macro_idx.intersection(ngrams_macro.index)
    micro_idx = micro_idx.intersection(ngrams_micro.index)

    print(f"\nAfter aligning with n-gram features:")
    print(f"  Macro-level: {len(macro_idx)} records (was {len(targets['y_macro_3way'])})")
    print(f"  Micro-level: {len(micro_idx)} records (was {len(targets['y_micro_ablaut'])})")

    # Macro-level models (3)
    for domain_name in ['has_suffix', 'has_mutation', '3way']:
        target_key = f'y_macro_{domain_name}'
        X = create_feature_matrix(df.loc[macro_idx], f'macro_{domain_name}', ngrams_macro.loc[macro_idx])
        y = targets[target_key].loc[macro_idx]
        save_features_and_targets(X, y, f'macro_{domain_name}')

    # Micro-level models (7)
    for domain_name in ['medial_a', 'final_a', 'final_vw', 'ablaut', 'insert_c', 'templatic', '8way']:
        target_key = f'y_micro_{domain_name}'
        X = create_feature_matrix(df.loc[micro_idx], f'micro_{domain_name}', ngrams_micro.loc[micro_idx])
        y = targets[target_key].loc[micro_idx]
        save_features_and_targets(X, y, f'micro_{domain_name}')

    print("\n" + "="*80)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*80)
    print(f"\nCreated features for 10 model domains:")
    print(f"  Macro-level: has_suffix, has_mutation, 3way")
    print(f"  Micro-level: medial_a, final_a, final_vw, ablaut, insert_c, templatic, 8way")
    print(f"\nOutput directory: {FEATURES_DIR}")
    print(f"  X_<domain>.csv - Feature matrices")
    print(f"  y_<domain>.csv - Target variables")
    print(f"  feature_metadata_<domain>.json - Metadata files")


if __name__ == '__main__':
    main()
