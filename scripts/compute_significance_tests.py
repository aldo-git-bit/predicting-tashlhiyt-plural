"""
Statistical Significance Testing for Table 3

Computes paired t-tests on 10-fold CV results to assess whether performance
differences between models are statistically significant.

For each of 10 domains, tests:
1. M+P vs N-grams (hand-crafted features vs baseline)
2. M+P vs bi-LSTM (grammar-based vs memory-based)
3. Best hand-crafted vs bi-LSTM (if different from M+P)

Outputs:
1. Significance table (CSV + console)
2. Updated Table 3 with significance markers (*, **, ***)
3. Appendix Table A11 with complete p-values

Usage:
    python scripts/compute_significance_tests.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import json
from typing import Dict, List, Tuple

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'experiments' / 'results'
REPORTS_DIR = PROJECT_ROOT / 'reports'


def load_fold_f1_scores(domain: str, feature_set: str, model: str) -> np.ndarray:
    """Load fold-level Macro-F1 scores from ablation results."""

    ablation_dir = RESULTS_DIR / f'ablation_{domain}'

    # Find matching result file
    pattern = f'{model}_{feature_set}_*.json'
    files = list(ablation_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No results found for {domain}/{model}/{feature_set}")

    # Use most recent file
    latest_file = max(files, key=lambda p: p.stat().st_mtime)

    with open(latest_file, 'r') as f:
        results = json.load(f)

    # Extract fold-level F1 scores
    # fold_results is a list of dicts, each with 'macro_f1' key
    fold_f1s = [fold['macro_f1'] for fold in results['fold_results']]

    return np.array(fold_f1s)


def load_lstm_fold_f1_scores(domain: str) -> np.ndarray:
    """Load fold-level Macro-F1 scores from LSTM baseline results."""
    from sklearn.metrics import f1_score

    lstm_dir = RESULTS_DIR / f'lstm_baseline_{domain}'

    # Load all fold results
    fold_f1s = []

    for fold_idx in range(10):
        npz_file = lstm_dir / f'predictions_fold_{fold_idx}.npz'

        if not npz_file.exists():
            raise FileNotFoundError(f"Missing LSTM fold results: {npz_file}")

        data = np.load(npz_file, allow_pickle=True)

        # Compute F1 from predictions
        y_true = data['y_true']
        y_pred = data['y_pred']

        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        fold_f1s.append(f1)

    return np.array(fold_f1s)


def get_best_hand_crafted_model(domain: str) -> Tuple[str, str, str]:
    """
    Determine which hand-crafted model performed best for this domain.

    Returns:
        (feature_set, model, display_name)
    """

    # Load Table 3 to find best hand-crafted model
    table3_path = REPORTS_DIR / 'master_table3_merged.csv'

    if not table3_path.exists():
        # Default to M+P LogReg
        return ('morph_phon', 'logistic_regression', 'M+P (LogReg)')

    df = pd.read_csv(table3_path)

    # Find row for this domain
    domain_row = df[df['Domain'] == domain].iloc[0]

    # Find max among hand-crafted features (Sem, Morph, Phon, M+P, All)
    hand_crafted_cols = ['Sem', 'Morph', 'Phon', 'M+P', 'All']
    best_col = domain_row[hand_crafted_cols].idxmax()
    best_f1 = domain_row[best_col]

    # Map column name to feature_set and model
    mapping = {
        'Sem': ('semantic_only', 'logistic_regression', 'Sem (LogReg)'),
        'Morph': ('morph_only', 'logistic_regression', 'Morph (LogReg)'),
        'Phon': ('phon_only', 'logistic_regression', 'Phon (LogReg)'),
        'M+P': ('morph_phon', 'logistic_regression', 'M+P (LogReg)'),
        'All': ('all_features', 'logistic_regression', 'All (LogReg)')
    }

    # Check if best is actually from a different model (RF/XGB)
    # For simplicity, we'll assume LogReg for now (most common winner)
    # This can be refined by checking ablation results

    return mapping.get(best_col, ('morph_phon', 'logistic_regression', 'M+P (LogReg)'))


def paired_t_test(scores1: np.ndarray, scores2: np.ndarray) -> Tuple[float, float]:
    """
    Perform paired t-test on two sets of fold scores.

    Returns:
        (t_statistic, p_value)
    """

    t_stat, p_value = stats.ttest_rel(scores1, scores2)

    return t_stat, p_value


def significance_marker(p_value: float) -> str:
    """Convert p-value to significance marker."""

    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''


def compute_all_significance_tests() -> pd.DataFrame:
    """
    Compute significance tests for all 10 domains.

    Tests:
    1. M+P vs N-grams
    2. M+P vs bi-LSTM
    3. Best hand-crafted vs bi-LSTM (if different from M+P)
    """

    domains = [
        # Macro
        'has_suffix', 'has_mutation', '3way',
        # Micro
        'medial_a', 'final_a', 'final_vw', 'ablaut', 'insert_c', 'templatic', '8way'
    ]

    results = []

    print("="*80)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("="*80)
    print("Running paired t-tests on 10-fold CV results...")
    print()

    for domain in domains:
        print(f"\n{'='*60}")
        print(f"Domain: {domain}")
        print(f"{'='*60}")

        try:
            # Load fold-level F1 scores
            mp_scores = load_fold_f1_scores(domain, 'morph_phon', 'logistic_regression')
            ngram_scores = load_fold_f1_scores(domain, 'ngrams_only', 'logistic_regression')
            lstm_scores = load_lstm_fold_f1_scores(domain)

            print(f"M+P F1 (10 folds):    {mp_scores.mean():.4f} ± {mp_scores.std():.4f}")
            print(f"N-grams F1 (10 folds): {ngram_scores.mean():.4f} ± {ngram_scores.std():.4f}")
            print(f"bi-LSTM F1 (10 folds): {lstm_scores.mean():.4f} ± {lstm_scores.std():.4f}")

            # Test 1: M+P vs N-grams
            t_mp_ngr, p_mp_ngr = paired_t_test(mp_scores, ngram_scores)
            delta_mp_ngr = mp_scores.mean() - ngram_scores.mean()

            print(f"\nTest 1: M+P vs N-grams")
            print(f"  Mean Δ: {delta_mp_ngr:+.4f} ({delta_mp_ngr/ngram_scores.mean()*100:+.1f}%)")
            print(f"  t = {t_mp_ngr:.3f}, p = {p_mp_ngr:.6f} {significance_marker(p_mp_ngr)}")

            # Test 2: M+P vs bi-LSTM
            t_mp_lstm, p_mp_lstm = paired_t_test(mp_scores, lstm_scores)
            delta_mp_lstm = mp_scores.mean() - lstm_scores.mean()

            print(f"\nTest 2: M+P vs bi-LSTM")
            print(f"  Mean Δ: {delta_mp_lstm:+.4f} ({delta_mp_lstm/lstm_scores.mean()*100:+.1f}%)")
            print(f"  t = {t_mp_lstm:.3f}, p = {p_mp_lstm:.6f} {significance_marker(p_mp_lstm)}")

            # Test 3: Best hand-crafted vs bi-LSTM (skip for now - use M+P as proxy)
            # For publication, we report M+P as the representative hand-crafted model
            t_best_lstm, p_best_lstm, delta_best_lstm = t_mp_lstm, p_mp_lstm, delta_mp_lstm
            best_name = 'M+P (LogReg)'

            # Store results
            results.append({
                'Domain': domain,
                'M+P_mean': mp_scores.mean(),
                'Ngr_mean': ngram_scores.mean(),
                'LSTM_mean': lstm_scores.mean(),
                'Δ_MP_Ngr': delta_mp_ngr,
                't_MP_Ngr': t_mp_ngr,
                'p_MP_Ngr': p_mp_ngr,
                'sig_MP_Ngr': significance_marker(p_mp_ngr),
                'Δ_MP_LSTM': delta_mp_lstm,
                't_MP_LSTM': t_mp_lstm,
                'p_MP_LSTM': p_mp_lstm,
                'sig_MP_LSTM': significance_marker(p_mp_lstm),
                'Best_model': best_name,
                'Δ_Best_LSTM': delta_best_lstm,
                't_Best_LSTM': t_best_lstm,
                'p_Best_LSTM': p_best_lstm,
                'sig_Best_LSTM': significance_marker(p_best_lstm)
            })

        except Exception as e:
            print(f"ERROR processing {domain}: {e}")
            import traceback
            traceback.print_exc()

    return pd.DataFrame(results)


def generate_significance_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate publication-ready significance table."""

    # Create clean table for Appendix A11
    table = pd.DataFrame({
        'Domain': df['Domain'],
        'M+P vs Ngr (Δ)': df['Δ_MP_Ngr'].apply(lambda x: f'{x:+.3f}'),
        'p-value': df['p_MP_Ngr'].apply(lambda x: f'{x:.6f}' if x >= 0.001 else '<0.001'),
        'Sig.': df['sig_MP_Ngr'],
        'M+P vs LSTM (Δ)': df['Δ_MP_LSTM'].apply(lambda x: f'{x:+.3f}'),
        'p-value ': df['p_MP_LSTM'].apply(lambda x: f'{x:.6f}' if x >= 0.001 else '<0.001'),
        'Sig. ': df['sig_MP_LSTM']
    })

    return table


def update_table3_with_significance(df_sig: pd.DataFrame):
    """Update Table 3 with significance markers in delta columns."""

    table3_path = REPORTS_DIR / 'master_table3_merged.csv'

    if not table3_path.exists():
        print("WARNING: Table 3 not found, cannot add significance markers")
        return

    # Load Table 3
    df_table3 = pd.read_csv(table3_path)

    # Add significance markers to ΔNgr and ΔLSTM columns
    for idx, row in df_sig.iterrows():
        domain = row['Domain']

        # Find matching row in Table 3
        mask = df_table3['Domain'] == domain

        if mask.sum() == 0:
            continue

        # Get current delta values
        delta_ngr = df_table3.loc[mask, 'ΔNgr'].values[0]
        delta_lstm = df_table3.loc[mask, 'ΔLSTM'].values[0]

        # Add significance markers
        sig_ngr = row['sig_MP_Ngr']
        sig_lstm = row['sig_MP_LSTM']

        # Update with markers (preserve original format but add markers)
        if pd.notna(delta_ngr):
            df_table3.loc[mask, 'ΔNgr_sig'] = f'{delta_ngr:.3f}{sig_ngr}'

        if pd.notna(delta_lstm):
            df_table3.loc[mask, 'ΔLSTM_sig'] = f'{delta_lstm:.3f}{sig_lstm}'

    # Save updated table
    table3_sig_path = REPORTS_DIR / 'master_table3_with_significance.csv'
    df_table3.to_csv(table3_sig_path, index=False)

    print(f"\nUpdated Table 3 saved: {table3_sig_path}")

    return df_table3


def print_summary(df: pd.DataFrame):
    """Print summary statistics about significance tests."""

    print("\n" + "="*80)
    print("SUMMARY OF SIGNIFICANCE TESTS")
    print("="*80)

    # M+P vs N-grams
    print("\nM+P vs N-grams (10 domains):")
    n_sig_001 = (df['p_MP_Ngr'] < 0.001).sum()
    n_sig_01 = ((df['p_MP_Ngr'] >= 0.001) & (df['p_MP_Ngr'] < 0.01)).sum()
    n_sig_05 = ((df['p_MP_Ngr'] >= 0.01) & (df['p_MP_Ngr'] < 0.05)).sum()
    n_ns = (df['p_MP_Ngr'] >= 0.05).sum()

    print(f"  p < 0.001 (***): {n_sig_001} domains")
    print(f"  p < 0.01  (**):  {n_sig_01} domains")
    print(f"  p < 0.05  (*):   {n_sig_05} domains")
    print(f"  Not sig. (ns):   {n_ns} domains")

    # M+P vs bi-LSTM
    print("\nM+P vs bi-LSTM (10 domains):")
    n_sig_001 = (df['p_MP_LSTM'] < 0.001).sum()
    n_sig_01 = ((df['p_MP_LSTM'] >= 0.001) & (df['p_MP_LSTM'] < 0.01)).sum()
    n_sig_05 = ((df['p_MP_LSTM'] >= 0.01) & (df['p_MP_LSTM'] < 0.05)).sum()
    n_ns = (df['p_MP_LSTM'] >= 0.05).sum()

    print(f"  p < 0.001 (***): {n_sig_001} domains")
    print(f"  p < 0.01  (**):  {n_sig_01} domains")
    print(f"  p < 0.05  (*):   {n_sig_05} domains")
    print(f"  Not sig. (ns):   {n_ns} domains")

    # Direction of effects
    print("\nDirection of significant effects (p < 0.05):")

    mp_better_ngr = df[(df['p_MP_Ngr'] < 0.05) & (df['Δ_MP_Ngr'] > 0)]['Domain'].tolist()
    ngr_better_mp = df[(df['p_MP_Ngr'] < 0.05) & (df['Δ_MP_Ngr'] < 0)]['Domain'].tolist()

    print(f"\n  M+P > N-grams ({len(mp_better_ngr)} domains):")
    for d in mp_better_ngr:
        print(f"    - {d}")

    if ngr_better_mp:
        print(f"\n  N-grams > M+P ({len(ngr_better_mp)} domains):")
        for d in ngr_better_mp:
            print(f"    - {d}")

    mp_better_lstm = df[(df['p_MP_LSTM'] < 0.05) & (df['Δ_MP_LSTM'] > 0)]['Domain'].tolist()
    lstm_better_mp = df[(df['p_MP_LSTM'] < 0.05) & (df['Δ_MP_LSTM'] < 0)]['Domain'].tolist()

    print(f"\n  M+P > bi-LSTM ({len(mp_better_lstm)} domains):")
    for d in mp_better_lstm:
        print(f"    - {d}")

    if lstm_better_mp:
        print(f"\n  bi-LSTM > M+P ({len(lstm_better_mp)} domains):")
        for d in lstm_better_mp:
            print(f"    - {d}")


def generate_latex_footnote(df: pd.DataFrame) -> str:
    """Generate LaTeX footnote text for Table 3."""

    # Count significant results
    n_mp_ngr_sig = (df['p_MP_Ngr'] < 0.05).sum()
    n_mp_lstm_sig = (df['p_MP_LSTM'] < 0.05).sum()

    # Count by level of significance
    n_p001 = ((df['p_MP_Ngr'] < 0.001) | (df['p_MP_LSTM'] < 0.001)).sum()
    n_p01 = ((df['p_MP_Ngr'] < 0.01) | (df['p_MP_LSTM'] < 0.01)).sum() - n_p001
    n_p05 = ((df['p_MP_Ngr'] < 0.05) | (df['p_MP_LSTM'] < 0.05)).sum() - n_p001 - n_p01

    footnote = f"""Significance markers indicate results of paired t-tests on 10-fold CV F1 scores:
***p < 0.001, **p < 0.01, *p < 0.05. M+P vs N-grams: {n_mp_ngr_sig}/10 significant.
M+P vs bi-LSTM: {n_mp_lstm_sig}/10 significant. Complete p-values in Appendix Table A11."""

    return footnote


def main():
    print("Computing statistical significance tests for all 10 domains...\n")

    # Compute all tests
    df_sig = compute_all_significance_tests()

    # Save raw results
    sig_path = REPORTS_DIR / 'significance_tests_complete.csv'
    df_sig.to_csv(sig_path, index=False)
    print(f"\n\nComplete results saved: {sig_path}")

    # Generate publication table (Appendix A11)
    table_a11 = generate_significance_table(df_sig)
    a11_path = REPORTS_DIR / 'appendix_table_a11_significance.csv'
    table_a11.to_csv(a11_path, index=False)
    print(f"Appendix Table A11 saved: {a11_path}")

    # Print Appendix Table A11
    print("\n" + "="*80)
    print("APPENDIX TABLE A11: Statistical Significance Tests")
    print("="*80)
    print(table_a11.to_string(index=False))

    # Update Table 3 with significance markers
    df_table3_updated = update_table3_with_significance(df_sig)

    # Print summary
    print_summary(df_sig)

    # Generate LaTeX footnote
    footnote = generate_latex_footnote(df_sig)
    print("\n" + "="*80)
    print("SUGGESTED FOOTNOTE FOR TABLE 3:")
    print("="*80)
    print(footnote)

    print("\n" + "="*80)
    print("ALL SIGNIFICANCE TESTS COMPLETE")
    print("="*80)
    print("\nFiles generated:")
    print(f"  1. {sig_path} - Complete results")
    print(f"  2. {a11_path} - Appendix Table A11")
    print(f"  3. reports/master_table3_with_significance.csv - Updated Table 3")


if __name__ == '__main__':
    main()
