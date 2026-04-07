#!/usr/bin/env python3
"""
Comprehensive test of phonological feature groupings for predicting mutation types.

This script tests expanded linguistic hypotheses about which phonological patterns
predict specific mutations, including:
- Binary groupings from LH patterns and foot structure
- Continuous count variables (with optimal binary split detection)
- Categorical groupings (multi-category analysis)
- Redundancy analysis between groupings

Usage:
    python test_phonological_groupings_comprehensive.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2_contingency
from itertools import combinations
import json

# ============================================================================
# Data Loading and Feature Engineering
# ============================================================================

def load_data():
    """Load the main dataset."""
    data_path = Path(__file__).parent.parent / 'data' / 'tash_nouns.csv'
    return pd.read_csv(data_path)

def calculate_mora_count(lh_pattern):
    """Calculate total mora count from LH pattern (H=2, L=1)."""
    if pd.isna(lh_pattern):
        return np.nan
    return sum(2 if char == 'H' else 1 if char == 'L' else 0 for char in lh_pattern)

def add_all_groupings(df):
    """Add all phonological grouping columns to dataframe."""

    # Clean whitespace from p_stem_sing_foot
    df['p_stem_sing_foot'] = df['p_stem_sing_foot'].str.strip()

    # === LH-based binary groupings ===
    df['p_LH_ends_L'] = (df['p_stem_sing_LH'].fillna('').str[-1] == 'L').astype(int)
    df['p_LH_initial_weight'] = (df['p_stem_sing_LH'].fillna('').str[0] == 'L').astype(int)
    df['p_LH_less_2_syllables'] = (df['p_stem_sing_LH'].fillna('').str.len() <= 2).astype(int)
    df['p_LH_all_light'] = (~df['p_stem_sing_LH'].fillna('').str.contains('H')).astype(int)
    df['p_LH_all_heavy'] = (~df['p_stem_sing_LH'].fillna('').str.contains('L')).astype(int)

    # === LH-based count variables ===
    df['p_LH_count_heavies'] = df['p_stem_sing_LH'].fillna('').apply(lambda x: x.count('H'))
    df['p_LH_count_moras'] = df['p_stem_sing_LH'].apply(calculate_mora_count)

    # === LH-based categorical ===
    # Extract last 2 syllables (or all if fewer than 2)
    df['p_LH_final_2'] = df['p_stem_sing_LH'].fillna('').apply(
        lambda x: x[-2:] if len(x) >= 2 else x
    )

    # === Foot-based binary groupings ===
    df['p_foot_residue_right'] = (df['p_stem_sing_foot'].fillna('').str[-1] == 'l').astype(int)
    df['p_foot_residue'] = df['p_stem_sing_foot'].fillna('').str.contains('l').astype(int)

    # === Foot-based count variables ===
    df['p_foot_count_feet'] = df['p_stem_sing_foot'].fillna('').apply(lambda x: x.count('F'))

    return df

# ============================================================================
# Testing Functions
# ============================================================================

def test_binary_grouping(df, target_mutation, grouping_col, grouping_name):
    """
    Test a binary phonological grouping for predicting a mutation.

    Args:
        df: DataFrame with data
        target_mutation: Mutation type to predict (e.g., 'Medial A')
        grouping_col: Name of column with binary grouping
        grouping_name: Name of the grouping for reporting

    Returns:
        dict with results
    """
    # Create binary target
    df['target'] = df['analysisInternalChanges'].fillna('').str.contains(target_mutation).astype(int)

    # Filter to valid cases (Internal/Mixed patterns only)
    valid_df = df[
        (df['analysisPluralPattern'].isin(['Internal', 'Mixed'])) &
        (df[grouping_col].notna())
    ].copy()

    if len(valid_df) == 0:
        return {'error': 'No valid cases'}

    # Cross-tabulation
    crosstab_counts = pd.crosstab(valid_df[grouping_col], valid_df['target'], margins=True)
    crosstab_pct = pd.crosstab(valid_df[grouping_col], valid_df['target'], normalize='index')

    # Chi-square test
    contingency = pd.crosstab(valid_df[grouping_col], valid_df['target'])
    if contingency.shape[0] > 1 and contingency.shape[1] > 1:
        chi2, p_value, dof, expected = chi2_contingency(contingency)
    else:
        chi2, p_value = None, None

    # Logistic regression
    X = valid_df[[grouping_col]].values
    y = valid_df['target'].values

    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X, y)

    odds_ratio = np.exp(lr.coef_[0][0])

    # Calculate effect size (percentage point difference)
    if crosstab_pct.shape[1] > 1 and crosstab_pct.shape[0] > 1:
        pct_group1 = crosstab_pct.iloc[1, 1]  # grouping=1, target=1
        pct_group0 = crosstab_pct.iloc[0, 1]  # grouping=0, target=1
        effect_size = abs(pct_group1 - pct_group0)
    else:
        effect_size = 0

    return {
        'mutation': target_mutation,
        'grouping': grouping_name,
        'type': 'binary',
        'n_cases': len(valid_df),
        'n_mutation': valid_df['target'].sum(),
        'pct_mutation': valid_df['target'].mean() * 100,
        'chi2': chi2,
        'p_value': p_value,
        'odds_ratio': odds_ratio,
        'effect_size': effect_size * 100,  # percentage points
    }

def test_continuous_grouping(df, target_mutation, grouping_col, grouping_name):
    """
    Test a continuous variable for predicting a mutation.
    Also finds optimal binary split point.

    Returns:
        dict with results including optimal split
    """
    # Create binary target
    df['target'] = df['analysisInternalChanges'].fillna('').str.contains(target_mutation).astype(int)

    # Filter to valid cases
    valid_df = df[
        (df['analysisPluralPattern'].isin(['Internal', 'Mixed'])) &
        (df[grouping_col].notna())
    ].copy()

    if len(valid_df) == 0:
        return {'error': 'No valid cases'}

    # Test continuous association
    X = valid_df[[grouping_col]].values
    y = valid_df['target'].values

    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X, y)

    # Chi-square test (using contingency table of all unique values)
    contingency = pd.crosstab(valid_df[grouping_col], valid_df['target'])
    if contingency.shape[0] > 1 and contingency.shape[1] > 1:
        chi2, p_value, dof, expected = chi2_contingency(contingency)
    else:
        chi2, p_value = None, None

    # Find optimal binary split
    unique_vals = sorted(valid_df[grouping_col].unique())
    best_split = None
    best_effect = 0
    best_or = None

    # Try each possible split point
    for split_val in unique_vals[:-1]:  # Don't split at max value
        valid_df['temp_binary'] = (valid_df[grouping_col] > split_val).astype(int)

        crosstab_pct = pd.crosstab(valid_df['temp_binary'], valid_df['target'], normalize='index')

        if crosstab_pct.shape[1] > 1 and crosstab_pct.shape[0] > 1:
            pct_group1 = crosstab_pct.iloc[1, 1]
            pct_group0 = crosstab_pct.iloc[0, 1]
            effect = abs(pct_group1 - pct_group0)

            # Also get OR for this split
            X_temp = valid_df[['temp_binary']].values
            lr_temp = LogisticRegression(random_state=42, max_iter=1000)
            lr_temp.fit(X_temp, y)
            or_temp = np.exp(lr_temp.coef_[0][0])

            if effect > best_effect:
                best_effect = effect
                best_split = split_val
                best_or = or_temp

    return {
        'mutation': target_mutation,
        'grouping': grouping_name,
        'type': 'continuous',
        'n_cases': len(valid_df),
        'n_mutation': valid_df['target'].sum(),
        'pct_mutation': valid_df['target'].mean() * 100,
        'chi2': chi2,
        'p_value': p_value,
        'optimal_split': best_split,
        'optimal_split_effect': best_effect * 100 if best_effect else 0,
        'optimal_split_or': best_or,
        'value_range': f"{valid_df[grouping_col].min()}-{valid_df[grouping_col].max()}",
    }

def test_categorical_grouping(df, target_mutation, grouping_col, grouping_name):
    """
    Test a categorical variable (multi-category) for predicting a mutation.

    Returns:
        dict with results including category-specific rates
    """
    # Create binary target
    df['target'] = df['analysisInternalChanges'].fillna('').str.contains(target_mutation).astype(int)

    # Filter to valid cases
    valid_df = df[
        (df['analysisPluralPattern'].isin(['Internal', 'Mixed'])) &
        (df[grouping_col].notna()) &
        (df[grouping_col] != '')
    ].copy()

    if len(valid_df) == 0:
        return {'error': 'No valid cases'}

    # Cross-tabulation
    crosstab_counts = pd.crosstab(valid_df[grouping_col], valid_df['target'], margins=True)
    crosstab_pct = pd.crosstab(valid_df[grouping_col], valid_df['target'], normalize='index')

    # Chi-square test
    contingency = pd.crosstab(valid_df[grouping_col], valid_df['target'])
    if contingency.shape[0] > 1 and contingency.shape[1] > 1:
        chi2, p_value, dof, expected = chi2_contingency(contingency)
    else:
        chi2, p_value = None, None

    # Get mutation rates by category
    category_rates = {}
    for cat in valid_df[grouping_col].unique():
        cat_df = valid_df[valid_df[grouping_col] == cat]
        rate = cat_df['target'].mean() * 100
        n = len(cat_df)
        category_rates[cat] = {'rate': rate, 'n': n}

    # Sort by rate
    category_rates = dict(sorted(category_rates.items(), key=lambda x: x[1]['rate'], reverse=True))

    return {
        'mutation': target_mutation,
        'grouping': grouping_name,
        'type': 'categorical',
        'n_cases': len(valid_df),
        'n_mutation': valid_df['target'].sum(),
        'pct_mutation': valid_df['target'].mean() * 100,
        'chi2': chi2,
        'p_value': p_value,
        'category_rates': category_rates,
        'n_categories': len(category_rates),
    }

# ============================================================================
# Redundancy Analysis
# ============================================================================

def analyze_redundancy(df, grouping_cols):
    """
    Analyze correlations between groupings to identify redundancies.

    Returns:
        DataFrame with pairwise correlations
    """
    valid_df = df[
        df['analysisPluralPattern'].isin(['Internal', 'Mixed'])
    ].copy()

    # Calculate correlations for binary/numeric groupings
    numeric_cols = [col for col in grouping_cols
                   if df[col].dtype in ['int64', 'float64']]

    if len(numeric_cols) < 2:
        return None

    corr_matrix = valid_df[numeric_cols].corr()

    # Get high correlations (|r| >= 0.7)
    high_corrs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) >= 0.7:
                high_corrs.append({
                    'grouping_1': corr_matrix.columns[i],
                    'grouping_2': corr_matrix.columns[j],
                    'correlation': corr
                })

    return pd.DataFrame(high_corrs)

# ============================================================================
# Main Analysis
# ============================================================================

def main():
    """Run comprehensive grouping tests."""
    print("Loading data and creating all groupings...")
    df = load_data()
    df = add_all_groupings(df)

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
    binary_groupings = [
        ('p_LH_ends_L', 'p_LH_ends_L'),
        ('p_LH_initial_weight', 'p_LH_initial_weight'),
        ('p_LH_less_2_syllables', 'p_LH_less_2_syllables'),
        ('p_LH_all_light', 'p_LH_all_light'),
        ('p_LH_all_heavy', 'p_LH_all_heavy'),
        ('p_foot_residue_right', 'p_foot_residue_right'),
        ('p_foot_residue', 'p_foot_residue'),
    ]

    continuous_groupings = [
        ('p_LH_count_heavies', 'p_LH_count_heavies'),
        ('p_LH_count_moras', 'p_LH_count_moras'),
        ('p_foot_count_feet', 'p_foot_count_feet'),
    ]

    categorical_groupings = [
        ('p_LH_final_2', 'p_LH_final_2'),
    ]

    print("\n" + "=" * 80)
    print("COMPREHENSIVE PHONOLOGICAL GROUPING HYPOTHESIS TESTS")
    print("=" * 80)
    print()

    # Collect all results
    all_results = []

    # ========================================================================
    # Test Binary Groupings
    # ========================================================================

    print("\n" + "=" * 80)
    print("PART 1: BINARY GROUPINGS")
    print("=" * 80)

    for mutation in mutations:
        print(f"\n{mutation}:")
        print("-" * 40)

        for grouping_col, grouping_name in binary_groupings:
            result = test_binary_grouping(df.copy(), mutation, grouping_col, grouping_name)

            if 'error' in result:
                continue

            all_results.append(result)

            # Print summary
            if result['p_value'] is not None and result['p_value'] < 0.05:
                sig_marker = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*"
            else:
                sig_marker = ""

            p_val_str = f"{result['p_value']:.4f}" if result['p_value'] is not None else "N/A"
            print(f"  {grouping_name:30s} | OR: {result['odds_ratio']:5.2f} | "
                  f"Effect: {result['effect_size']:5.1f}pp | "
                  f"p: {p_val_str:>7s} {sig_marker}")

    # ========================================================================
    # Test Continuous Groupings
    # ========================================================================

    print("\n\n" + "=" * 80)
    print("PART 2: CONTINUOUS GROUPINGS (with optimal binary splits)")
    print("=" * 80)

    for mutation in mutations:
        print(f"\n{mutation}:")
        print("-" * 40)

        for grouping_col, grouping_name in continuous_groupings:
            result = test_continuous_grouping(df.copy(), mutation, grouping_col, grouping_name)

            if 'error' in result:
                continue

            all_results.append(result)

            # Print summary
            if result['p_value'] is not None and result['p_value'] < 0.05:
                sig_marker = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*"
            else:
                sig_marker = ""

            p_val_str = f"{result['p_value']:.4f}" if result['p_value'] is not None else "N/A"
            split_info = f"Split>{result['optimal_split']} (Effect: {result['optimal_split_effect']:.1f}pp, OR: {result['optimal_split_or']:.2f})" if result['optimal_split'] is not None else "N/A"

            print(f"  {grouping_name:30s} | Range: {result['value_range']:>6s} | "
                  f"p: {p_val_str:>7s} {sig_marker}")
            print(f"  {'':30s}   Optimal split: {split_info}")

    # ========================================================================
    # Test Categorical Groupings
    # ========================================================================

    print("\n\n" + "=" * 80)
    print("PART 3: CATEGORICAL GROUPINGS")
    print("=" * 80)

    for mutation in mutations:
        print(f"\n{mutation}:")
        print("-" * 40)

        for grouping_col, grouping_name in categorical_groupings:
            result = test_categorical_grouping(df.copy(), mutation, grouping_col, grouping_name)

            if 'error' in result:
                continue

            all_results.append(result)

            # Print summary
            if result['p_value'] is not None and result['p_value'] < 0.05:
                sig_marker = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*"
            else:
                sig_marker = ""

            p_val_str = f"{result['p_value']:.4f}" if result['p_value'] is not None else "N/A"
            print(f"  {grouping_name:30s} | Categories: {result['n_categories']:2d} | "
                  f"p: {p_val_str:>7s} {sig_marker}")

            # Print top categories
            print(f"  {'':30s}   Top categories by mutation rate:")
            for cat, data in list(result['category_rates'].items())[:3]:
                print(f"  {'':30s}     {cat:>4s}: {data['rate']:5.1f}% (n={data['n']})")

    # ========================================================================
    # Redundancy Analysis
    # ========================================================================

    print("\n\n" + "=" * 80)
    print("PART 4: REDUNDANCY ANALYSIS")
    print("=" * 80)

    all_grouping_cols = [g[0] for g in binary_groupings + continuous_groupings]
    redundancy_df = analyze_redundancy(df, all_grouping_cols)

    if redundancy_df is not None and len(redundancy_df) > 0:
        print("\nHigh correlations (|r| >= 0.7):")
        print("-" * 80)
        for _, row in redundancy_df.iterrows():
            print(f"  {row['grouping_1']:30s} ↔ {row['grouping_2']:30s} | r = {row['correlation']:5.2f}")
    else:
        print("\nNo high correlations found (|r| >= 0.7)")

    # ========================================================================
    # Summary of Significant Results
    # ========================================================================

    print("\n\n" + "=" * 80)
    print("SUMMARY: SIGNIFICANT ASSOCIATIONS (p < 0.05)")
    print("=" * 80)

    sig_results = [r for r in all_results
                   if r.get('p_value') is not None and r['p_value'] < 0.05]

    print(f"\nTotal significant associations: {len(sig_results)}")

    # Group by mutation
    for mutation in mutations:
        mut_results = [r for r in sig_results if r['mutation'] == mutation]
        if mut_results:
            print(f"\n{mutation} ({len(mut_results)} significant):")
            for r in sorted(mut_results, key=lambda x: x.get('effect_size', x.get('optimal_split_effect', 0)), reverse=True):
                if r['type'] == 'binary':
                    print(f"  {r['grouping']:30s} | OR: {r['odds_ratio']:5.2f} | Effect: {r['effect_size']:5.1f}pp")
                elif r['type'] == 'continuous':
                    print(f"  {r['grouping']:30s} | Optimal split >{r['optimal_split']}: {r['optimal_split_effect']:5.1f}pp (OR: {r['optimal_split_or']:.2f})")
                elif r['type'] == 'categorical':
                    print(f"  {r['grouping']:30s} | {r['n_categories']} categories (chi2={r['chi2']:.1f})")

    # ========================================================================
    # Save Results
    # ========================================================================

    output_path = Path(__file__).parent.parent / 'reports' / 'phonological_groupings_comprehensive_results.json'

    # Convert results to JSON-serializable format
    json_results = []
    for r in all_results:
        json_r = r.copy()
        # Convert category_rates dict if present
        if 'category_rates' in json_r:
            json_r['category_rates'] = {k: {'rate': float(v['rate']), 'n': int(v['n'])}
                                       for k, v in json_r['category_rates'].items()}
        # Convert numpy types
        for key in json_r:
            if isinstance(json_r[key], (np.integer, np.floating)):
                json_r[key] = float(json_r[key])
        json_results.append(json_r)

    with open(output_path, 'w') as f:
        json.dump({
            'results': json_results,
            'redundancy': redundancy_df.to_dict('records') if redundancy_df is not None else []
        }, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")

if __name__ == '__main__':
    main()
