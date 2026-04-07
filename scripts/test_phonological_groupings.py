#!/usr/bin/env python3
"""
Test phonological feature groupings for predicting mutation types.

This script tests linguistic hypotheses about which phonological patterns
predict specific mutations, helping inform feature engineering decisions.

Usage:
    python test_phonological_groupings.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy.stats import chi2_contingency

def load_data():
    """Load the main dataset."""
    data_path = Path(__file__).parent.parent / 'data' / 'tash_nouns.csv'
    return pd.read_csv(data_path)

def test_grouping(df, target_mutation, grouping_func, grouping_name):
    """
    Test a phonological grouping strategy for predicting a mutation.

    Args:
        df: DataFrame with data
        target_mutation: Mutation type to predict (e.g., 'Medial A')
        grouping_func: Function that takes df and returns binary grouping column
        grouping_name: Name of the grouping for reporting

    Returns:
        dict with results
    """
    # Create binary target
    df['target'] = df['analysisInternalChanges'].fillna('').str.contains(target_mutation).astype(int)

    # Apply grouping function
    df['grouping'] = grouping_func(df)

    # Filter to valid cases (Internal/Mixed patterns only)
    valid_df = df[
        (df['analysisPluralPattern'].isin(['Internal', 'Mixed'])) &
        (df['grouping'].notna())
    ].copy()

    if len(valid_df) == 0:
        return {'error': 'No valid cases'}

    # Cross-tabulation
    crosstab_counts = pd.crosstab(valid_df['grouping'], valid_df['target'], margins=True)
    crosstab_pct = pd.crosstab(valid_df['grouping'], valid_df['target'], normalize='index')

    # Chi-square test
    contingency = pd.crosstab(valid_df['grouping'], valid_df['target'])
    if contingency.shape[0] > 1 and contingency.shape[1] > 1:
        chi2, p_value, dof, expected = chi2_contingency(contingency)
    else:
        chi2, p_value = None, None

    # Logistic regression
    X = valid_df[['grouping']].values
    y = valid_df['target'].values

    lr = LogisticRegression(random_state=42)
    lr.fit(X, y)

    odds_ratio = np.exp(lr.coef_[0][0])

    # Calculate effect size (percentage point difference)
    pct_group1 = crosstab_pct.iloc[0, 1] if crosstab_pct.shape[1] > 1 else 0
    pct_group0 = crosstab_pct.iloc[1, 1] if crosstab_pct.shape[1] > 1 and crosstab_pct.shape[0] > 1 else 0
    effect_size = abs(pct_group1 - pct_group0)

    return {
        'mutation': target_mutation,
        'grouping': grouping_name,
        'n_cases': len(valid_df),
        'n_mutation': valid_df['target'].sum(),
        'pct_mutation': valid_df['target'].mean() * 100,
        'chi2': chi2,
        'p_value': p_value,
        'odds_ratio': odds_ratio,
        'effect_size': effect_size * 100,  # percentage points
        'crosstab': crosstab_counts,
        'crosstab_pct': crosstab_pct
    }

# ============================================================================
# Define Grouping Functions
# ============================================================================

def lh_ends_in_L(df):
    """Grouping: LH pattern ends in L (vs H)."""
    return (df['p_stem_sing_LH'].fillna('').str[-1] == 'L').astype(int)

def lh_starts_with_L(df):
    """Grouping: LH pattern starts with L (vs H)."""
    return (df['p_stem_sing_LH'].fillna('').str[0] == 'L').astype(int)

def lh_has_final_LL(df):
    """Grouping: LH pattern ends with LL (vs not)."""
    return df['p_stem_sing_LH'].fillna('').str.endswith('LL').astype(int)

def lh_all_light(df):
    """Grouping: All light syllables (vs has at least one H)."""
    return (~df['p_stem_sing_LH'].fillna('').str.contains('H')).astype(int)

def lh_all_heavy(df):
    """Grouping: All heavy syllables (vs has at least one L)."""
    return (~df['p_stem_sing_LH'].fillna('').str.contains('L')).astype(int)

def lh_length_short(df):
    """Grouping: Short stems (1-2 syllables vs 3+)."""
    return (df['p_stem_sing_LH'].fillna('').str.len() <= 2).astype(int)

def foot_unparsed_final(df):
    """Grouping: Foot structure ends in unparsed 'l' (vs 'F')."""
    return (df['p_stem_sing_foot'].fillna('').str.strip().str[-1] == 'l').astype(int)

# ============================================================================
# Main Analysis
# ============================================================================

def main():
    """Run comprehensive grouping tests."""
    df = load_data()

    # Define mutations to test
    mutations = [
        'Medial A',
        'Final A',
        'Final Vw',
        'Ablaut',
        'Insert C',
        'Templatic'
    ]

    # Define groupings to test
    groupings = [
        (lh_ends_in_L, 'LH ends in L'),
        (lh_starts_with_L, 'LH starts with L'),
        (lh_has_final_LL, 'LH ends in LL'),
        (lh_all_light, 'All Light syllables'),
        (lh_all_heavy, 'All Heavy syllables'),
        (lh_length_short, 'Short stem (≤2 syllables)'),
        (foot_unparsed_final, 'Unparsed final syllable (l)')
    ]

    print("=" * 80)
    print("PHONOLOGICAL GROUPING HYPOTHESIS TESTS")
    print("=" * 80)
    print()
    print("Testing which phonological groupings predict specific mutation types")
    print()

    results = []

    for mutation in mutations:
        print(f"\n{'=' * 80}")
        print(f"MUTATION: {mutation}")
        print(f"{'=' * 80}\n")

        for grouping_func, grouping_name in groupings:
            result = test_grouping(df.copy(), mutation, grouping_func, grouping_name)

            if 'error' in result:
                continue

            results.append(result)

            # Print summary
            if result['p_value'] is not None and result['p_value'] < 0.05:
                sig_marker = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*"
            else:
                sig_marker = ""

            p_val_str = f"{result['p_value']:.4f}" if result['p_value'] is not None else "N/A"
            print(f"{grouping_name:35s} | OR: {result['odds_ratio']:5.2f} | "
                  f"Effect: {result['effect_size']:5.1f}pp | "
                  f"p: {p_val_str:>7s} {sig_marker}")

    # Summary of strongest effects
    print(f"\n{'=' * 80}")
    print("STRONGEST ASSOCIATIONS (Effect size ≥ 15pp and p < 0.05)")
    print(f"{'=' * 80}\n")

    strong_results = [r for r in results
                     if r.get('p_value') is not None
                     and r['p_value'] < 0.05
                     and r['effect_size'] >= 15]

    strong_results.sort(key=lambda x: x['effect_size'], reverse=True)

    print(f"{'Mutation':15s} | {'Grouping':35s} | {'OR':>6s} | {'Effect':>8s} | {'p-value':>8s}")
    print("-" * 80)

    for r in strong_results:
        print(f"{r['mutation']:15s} | {r['grouping']:35s} | {r['odds_ratio']:6.2f} | "
              f"{r['effect_size']:6.1f}pp | {r['p_value']:.4f}")

    print(f"\nTotal strong associations found: {len(strong_results)}")
    print("\nNote: Effect size = percentage point difference in mutation rate between groups")

if __name__ == '__main__':
    main()
