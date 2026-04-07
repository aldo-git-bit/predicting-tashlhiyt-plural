#!/usr/bin/env python3
"""
Generate master publication tables for reporting.

Table 1: Performance comparison across feature sets and baselines
Table 2: Residual lexical idiosyncrasy analysis

Author: Claude Code
Date: January 3, 2026
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

# Feature sets for Table 1
FEATURE_SETS = ['semantic_only', 'morph_only', 'phon_only', 'morph_phon', 'all_features', 'ngrams_only']
FEATURE_LABELS = {
    'semantic_only': 'Sem',
    'morph_only': 'Morph',
    'phon_only': 'Phon',
    'morph_phon': 'M+P',
    'all_features': 'All',
    'ngrams_only': 'Ngr'
}

# Domain display names
DOMAIN_LABELS = {
    'has_suffix': 'Has Suffix',
    'has_mutation': 'Has Mutation',
    '3way': '3-way',
    'medial_a': 'Medial A',
    'final_a': 'Final A',
    'final_vw': 'Final V/W',
    'ablaut': 'Ablaut',
    'insert_c': 'Insert C',
    'templatic': 'Templatic',
    '8way': '8-way'
}


def load_ablation_results(domain):
    """Load latest ablation results for a domain."""
    results_dir = PROJECT_ROOT / 'experiments' / 'results' / f'ablation_{domain}'
    summary_files = sorted(results_dir.glob('ablation_summary_*.csv'))

    if not summary_files:
        print(f"Warning: No ablation results found for {domain}")
        return None

    latest = summary_files[-1]
    df = pd.read_csv(latest)

    # Filter to Logistic Regression only
    df_lr = df[df['model'] == 'logistic_regression'].copy()

    return df_lr


def load_lstm_results(domain):
    """Load LSTM results for a domain."""
    results_dir = PROJECT_ROOT / 'experiments' / 'results' / f'lstm_baseline_{domain}'
    summary_file = results_dir / 'lstm_summary.csv'

    if not summary_file.exists():
        print(f"Warning: No LSTM results found for {domain}")
        return None

    df = pd.read_csv(summary_file)
    return df


def load_error_overlap(domain):
    """Load error overlap results for residual analysis."""
    error_file = PROJECT_ROOT / 'reports' / 'lstm_residuals' / f'error_overlap_{domain}.csv'

    if not error_file.exists():
        print(f"Warning: No error overlap file found for {domain}")
        return None

    df = pd.read_csv(error_file)
    return df


def load_confidence_analysis(domain):
    """Load confidence analysis for severity calculation."""
    conf_file = PROJECT_ROOT / 'reports' / 'lstm_residuals' / f'confidence_analysis_{domain}.csv'

    if not conf_file.exists():
        print(f"Warning: No confidence analysis file found for {domain}")
        return None

    df = pd.read_csv(conf_file)
    return df


def compute_severity(domain, conf_df, total_n):
    """
    Compute idiosyncrasy severity: % of all forms that are high-confidence exceptions.

    High-confidence exception = Both models fail AND mean confidence >= 0.8
    """
    if conf_df is None:
        return np.nan

    # Identify Both Fail cases
    # LSTM fails: lstm_prediction != true_label
    # MP fails: mp_prediction != true_label
    both_fail = (conf_df['lstm_prediction'] != conf_df['true_label']) & \
                (conf_df['mp_prediction'] != conf_df['true_label'])

    # Compute mean confidence for each case
    # Handle NaN in mp_confidence (some domains might not have MP predictions for all cases)
    mean_confidence = (conf_df['lstm_confidence'] + conf_df['mp_confidence'].fillna(0)) / 2

    # High confidence = mean >= 0.8
    high_conf = mean_confidence >= 0.8

    # Both fail AND high confidence
    both_fail_high_conf = both_fail & high_conf

    severity = (both_fail_high_conf.sum() / total_n) * 100
    return severity


def generate_table1():
    """Generate Table 1: Performance Comparison."""

    rows = []

    for domain in ALL_DOMAINS:
        ablation_df = load_ablation_results(domain)
        lstm_df = load_lstm_results(domain)

        if ablation_df is None:
            print(f"Skipping {domain} - no ablation data")
            continue

        row = {'Domain': DOMAIN_LABELS[domain]}

        # Extract F1 for each feature set (Logistic Regression)
        for feat_set in FEATURE_SETS[:5]:  # Exclude ngrams_only for now
            feat_data = ablation_df[ablation_df['feature_set'] == feat_set]
            if len(feat_data) > 0:
                f1 = feat_data.iloc[0]['macro_f1_mean']
                row[FEATURE_LABELS[feat_set]] = f1
            else:
                row[FEATURE_LABELS[feat_set]] = np.nan

        # N-gram baseline (from ablation)
        ngram_data = ablation_df[ablation_df['feature_set'] == 'ngrams_only']
        if len(ngram_data) > 0:
            row['Ngr'] = ngram_data.iloc[0]['macro_f1_mean']
        else:
            row['Ngr'] = np.nan

        # LSTM baseline
        if lstm_df is not None and len(lstm_df) > 0:
            row['LSTM'] = lstm_df.iloc[0]['macro_f1_mean']
        else:
            row['LSTM'] = np.nan

        # Compute Best among hand-crafted features
        hand_crafted = [row.get(FEATURE_LABELS[fs], np.nan) for fs in FEATURE_SETS[:5]]
        best_f1 = np.nanmax(hand_crafted)
        row['Best'] = best_f1

        # Compute deltas
        row['ΔNgr'] = best_f1 - row.get('Ngr', np.nan)
        row['ΔLSTM'] = best_f1 - row.get('LSTM', np.nan)

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Reorder columns
    cols = ['Domain', 'Sem', 'Morph', 'Phon', 'M+P', 'All', 'Ngr', 'LSTM', 'ΔNgr', 'ΔLSTM']
    df = df[cols]

    # Add section headers
    macro_header = pd.DataFrame([{
        'Domain': 'MACRO-LEVEL',
        'Sem': np.nan, 'Morph': np.nan, 'Phon': np.nan, 'M+P': np.nan, 'All': np.nan,
        'Ngr': np.nan, 'LSTM': np.nan, 'ΔNgr': np.nan, 'ΔLSTM': np.nan
    }])

    micro_header = pd.DataFrame([{
        'Domain': 'MICRO-LEVEL',
        'Sem': np.nan, 'Morph': np.nan, 'Phon': np.nan, 'M+P': np.nan, 'All': np.nan,
        'Ngr': np.nan, 'LSTM': np.nan, 'ΔNgr': np.nan, 'ΔLSTM': np.nan
    }])

    # Insert headers
    df_macro = df[df['Domain'].isin([DOMAIN_LABELS[d] for d in MACRO_DOMAINS])].copy()
    df_micro = df[df['Domain'].isin([DOMAIN_LABELS[d] for d in MICRO_DOMAINS])].copy()

    df_final = pd.concat([macro_header, df_macro, micro_header, df_micro], ignore_index=True)

    return df_final


def generate_table2():
    """Generate Table 2: Residual Analysis."""

    rows = []

    for domain in ALL_DOMAINS:
        ablation_df = load_ablation_results(domain)
        lstm_df = load_lstm_results(domain)
        overlap_df = load_error_overlap(domain)
        conf_df = load_confidence_analysis(domain)

        if ablation_df is None or lstm_df is None or overlap_df is None:
            print(f"Skipping {domain} - missing data")
            continue

        row = {'Domain': DOMAIN_LABELS[domain]}

        # Get Morph+Phon F1
        mp_data = ablation_df[ablation_df['feature_set'] == 'morph_phon']
        if len(mp_data) > 0:
            mp_f1 = mp_data.iloc[0]['macro_f1_mean']
        else:
            mp_f1 = np.nan

        # Get LSTM F1
        lstm_f1 = lstm_df.iloc[0]['macro_f1_mean'] if len(lstm_df) > 0 else np.nan

        # Computational Ceiling
        row['Ceil'] = max(mp_f1, lstm_f1)

        # Extract error overlap percentages
        both_fail_row = overlap_df[overlap_df['Category'] == 'Both Models Fail']
        lstm_only_row = overlap_df[overlap_df['Category'] == 'LSTM Only']
        mp_only_row = overlap_df[overlap_df['Category'] == 'Morph+Phon Only']

        if len(both_fail_row) > 0:
            irr_err = both_fail_row.iloc[0]['Percentage']
            row['IrrErr%'] = irr_err
            row['Learn%'] = 100 - irr_err
        else:
            row['IrrErr%'] = np.nan
            row['Learn%'] = np.nan

        # Complementarity
        if len(lstm_only_row) > 0 and len(mp_only_row) > 0 and row['Learn%'] > 0:
            lstm_only_pct = lstm_only_row.iloc[0]['Percentage']
            mp_only_pct = mp_only_row.iloc[0]['Percentage']
            compl = (lstm_only_pct + mp_only_pct) / row['Learn%']
            row['Compl'] = compl
        else:
            row['Compl'] = np.nan

        # Severity (need total n from overlap)
        total_row = overlap_df[overlap_df['Category'] == 'Total']
        if len(total_row) > 0:
            total_n = total_row.iloc[0]['Count']
            row['Sev%'] = compute_severity(domain, conf_df, total_n)
        else:
            row['Sev%'] = np.nan

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Reorder columns
    cols = ['Domain', 'Ceil', 'IrrErr%', 'Learn%', 'Compl', 'Sev%']
    df = df[cols]

    # Add section headers
    macro_header = pd.DataFrame([{
        'Domain': 'MACRO-LEVEL',
        'Ceil': np.nan, 'IrrErr%': np.nan, 'Learn%': np.nan, 'Compl': np.nan, 'Sev%': np.nan
    }])

    micro_header = pd.DataFrame([{
        'Domain': 'MICRO-LEVEL',
        'Ceil': np.nan, 'IrrErr%': np.nan, 'Learn%': np.nan, 'Compl': np.nan, 'Sev%': np.nan
    }])

    # Insert headers
    df_macro = df[df['Domain'].isin([DOMAIN_LABELS[d] for d in MACRO_DOMAINS])].copy()
    df_micro = df[df['Domain'].isin([DOMAIN_LABELS[d] for d in MICRO_DOMAINS])].copy()

    df_final = pd.concat([macro_header, df_macro, micro_header, df_micro], ignore_index=True)

    return df_final


def generate_table3():
    """
    Generate Table 3: Merged Performance and Residual Analysis.

    Standardizes on M+P as the reference model for deltas and residual analysis.
    Three-decimal precision for ACL standards.

    Two-level column headers:
    - Hand-Crafted Feature Models: Sem, Morph, Phon, M+P, All
    - Baseline Comparisons: Ngr, LSTM, ΔNgr, ΔLSTM
    - Residual Analysis: Ceil, IrrErr%, Compl, Sev%
    """
    rows = []

    for domain in ALL_DOMAINS:
        ablation_df = load_ablation_results(domain)
        lstm_df = load_lstm_results(domain)
        overlap_df = load_error_overlap(domain)
        conf_df = load_confidence_analysis(domain)

        if ablation_df is None:
            print(f"Skipping {domain} - no ablation data")
            continue

        row = {'Domain': DOMAIN_LABELS[domain]}

        # === HAND-CRAFTED FEATURE MODELS ===
        for feat_set in FEATURE_SETS[:5]:  # Sem, Morph, Phon, M+P, All
            feat_data = ablation_df[ablation_df['feature_set'] == feat_set]
            if len(feat_data) > 0:
                f1 = feat_data.iloc[0]['macro_f1_mean']
                row[FEATURE_LABELS[feat_set]] = f1
            else:
                row[FEATURE_LABELS[feat_set]] = np.nan

        # Get M+P F1 for standardization
        mp_f1 = row.get('M+P', np.nan)

        # === BASELINE COMPARISONS ===
        # N-gram baseline
        ngram_data = ablation_df[ablation_df['feature_set'] == 'ngrams_only']
        if len(ngram_data) > 0:
            row['Ngr'] = ngram_data.iloc[0]['macro_f1_mean']
        else:
            row['Ngr'] = np.nan

        # LSTM baseline
        if lstm_df is not None and len(lstm_df) > 0:
            lstm_f1 = lstm_df.iloc[0]['macro_f1_mean']
            row['LSTM'] = lstm_f1
        else:
            lstm_f1 = np.nan
            row['LSTM'] = np.nan

        # Deltas (standardized on M+P)
        row['ΔNgr'] = mp_f1 - row.get('Ngr', np.nan)
        row['ΔLSTM'] = mp_f1 - lstm_f1

        # === RESIDUAL ANALYSIS ===
        # Computational Ceiling (max of M+P and LSTM)
        row['Ceil'] = max(mp_f1, lstm_f1) if not (np.isnan(mp_f1) or np.isnan(lstm_f1)) else np.nan

        # Extract error overlap percentages
        if overlap_df is not None:
            both_fail_row = overlap_df[overlap_df['Category'] == 'Both Models Fail']
            lstm_only_row = overlap_df[overlap_df['Category'] == 'LSTM Only']
            mp_only_row = overlap_df[overlap_df['Category'] == 'Morph+Phon Only']

            if len(both_fail_row) > 0:
                irr_err = both_fail_row.iloc[0]['Percentage']
                row['IrrErr%'] = irr_err
            else:
                row['IrrErr%'] = np.nan

            # Complementarity
            if len(lstm_only_row) > 0 and len(mp_only_row) > 0 and not np.isnan(row.get('IrrErr%', np.nan)):
                lstm_only_pct = lstm_only_row.iloc[0]['Percentage']
                mp_only_pct = mp_only_row.iloc[0]['Percentage']
                learn_pct = 100 - row['IrrErr%']
                if learn_pct > 0:
                    compl = (lstm_only_pct + mp_only_pct) / learn_pct
                    row['Compl'] = compl
                else:
                    row['Compl'] = np.nan
            else:
                row['Compl'] = np.nan

            # Severity
            total_row = overlap_df[overlap_df['Category'] == 'Total']
            if len(total_row) > 0:
                total_n = total_row.iloc[0]['Count']
                row['Sev%'] = compute_severity(domain, conf_df, total_n)
            else:
                row['Sev%'] = np.nan
        else:
            row['IrrErr%'] = np.nan
            row['Compl'] = np.nan
            row['Sev%'] = np.nan

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Reorder columns
    cols = ['Domain',
            'Sem', 'Morph', 'Phon', 'M+P', 'All',  # Hand-crafted
            'Ngr', 'LSTM', 'ΔNgr', 'ΔLSTM',        # Baseline Comparisons
            'Ceil', 'IrrErr%', 'Compl', 'Sev%']    # Residual Analysis
    df = df[cols]

    # Add section headers
    macro_header = pd.DataFrame([{
        'Domain': 'MACRO-LEVEL',
        'Sem': np.nan, 'Morph': np.nan, 'Phon': np.nan, 'M+P': np.nan, 'All': np.nan,
        'Ngr': np.nan, 'LSTM': np.nan, 'ΔNgr': np.nan, 'ΔLSTM': np.nan,
        'Ceil': np.nan, 'IrrErr%': np.nan, 'Compl': np.nan, 'Sev%': np.nan
    }])

    micro_header = pd.DataFrame([{
        'Domain': 'MICRO-LEVEL',
        'Sem': np.nan, 'Morph': np.nan, 'Phon': np.nan, 'M+P': np.nan, 'All': np.nan,
        'Ngr': np.nan, 'LSTM': np.nan, 'ΔNgr': np.nan, 'ΔLSTM': np.nan,
        'Ceil': np.nan, 'IrrErr%': np.nan, 'Compl': np.nan, 'Sev%': np.nan
    }])

    # Insert headers
    df_macro = df[df['Domain'].isin([DOMAIN_LABELS[d] for d in MACRO_DOMAINS])].copy()
    df_micro = df[df['Domain'].isin([DOMAIN_LABELS[d] for d in MICRO_DOMAINS])].copy()

    df_final = pd.concat([macro_header, df_macro, micro_header, df_micro], ignore_index=True)

    return df_final


def main():
    """Generate all tables and save to CSV."""

    print("Generating Table 1: Performance Comparison...")
    table1 = generate_table1()

    print("Generating Table 2: Residual Analysis...")
    table2 = generate_table2()

    print("Generating Table 3: Merged Performance and Residual Analysis...")
    table3 = generate_table3()

    # Save to reports directory
    output_dir = PROJECT_ROOT / 'reports'
    output_dir.mkdir(exist_ok=True)

    table1_path = output_dir / 'master_table1_performance.csv'
    table2_path = output_dir / 'master_table2_residual.csv'
    table3_path = output_dir / 'master_table3_merged.csv'

    table1.to_csv(table1_path, index=False)
    table2.to_csv(table2_path, index=False)
    table3.to_csv(table3_path, index=False, float_format='%.3f')

    print(f"\nTable 1 saved to: {table1_path}")
    print(f"Table 2 saved to: {table2_path}")
    print(f"Table 3 saved to: {table3_path}")

    print("\n=== Table 1 Preview ===")
    print(table1.to_string(index=False))

    print("\n=== Table 2 Preview ===")
    print(table2.to_string(index=False))

    print("\n=== Table 3 Preview ===")
    print(table3.to_string(index=False, float_format=lambda x: f'{x:.3f}'))


if __name__ == '__main__':
    main()
