#!/usr/bin/env python3
"""
Residual Analysis: LSTM Baseline vs Morph+Phon Features

Analyzes error patterns and identifies lexical idiosyncrasies by comparing:
1. LSTM baseline predictions
2. Morph+Phon ablation predictions

Generates:
- Confidence analysis (high/medium/low confidence predictions)
- Error overlap analysis (LSTM-only errors, Features-only errors, Both fail)
- Idiosyncrasy rankings (forms hardest to predict)
- Venn diagrams of error patterns

Usage:
    python scripts/analyze_lstm_residuals.py --domain has_suffix
    python scripts/analyze_lstm_residuals.py --domain has_mutation
    python scripts/analyze_lstm_residuals.py --domain 3way
    python scripts/analyze_lstm_residuals.py --domain ablaut
    python scripts/analyze_lstm_residuals.py --domain medial_a
    (... and 5 more micro domains: final_a, final_vw, insert_c, templatic, 8way)
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

def load_lstm_predictions(domain):
    """Load LSTM baseline predictions across all folds."""
    lstm_dir = Path('experiments/results') / f'lstm_baseline_{domain}'

    all_predictions = []
    all_probabilities = []
    all_true_labels = []
    all_record_ids = []

    for fold in range(10):
        pred_file = lstm_dir / f'predictions_fold_{fold}.npz'
        if not pred_file.exists():
            print(f"Warning: {pred_file} not found")
            continue

        # Load .npz file
        data = np.load(pred_file, allow_pickle=True)

        # Extract data
        all_record_ids.extend(data['record_ids'].tolist())
        all_predictions.extend(data['y_pred'].tolist())
        all_probabilities.extend(data['y_pred_proba'].tolist())
        all_true_labels.extend(data['y_true'].tolist())

    return {
        'record_ids': all_record_ids,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'true_labels': all_true_labels
    }

def load_morph_phon_predictions(domain):
    """Load Morph+Phon ablation predictions (Logistic Regression)."""
    ablation_dir = Path('experiments/results') / f'ablation_{domain}'

    # Find the latest Morph+Phon LogReg experiment (by timestamp)
    morph_phon_files = list(ablation_dir.glob('logistic_regression_morph_phon_*.json'))

    if not morph_phon_files:
        raise FileNotFoundError(f"Morph+Phon LogReg results not found in {ablation_dir}")

    # Get the latest file (by modification time)
    morph_phon_file = max(morph_phon_files, key=lambda p: p.stat().st_mtime)
    print(f"  Loading: {morph_phon_file.name}")

    with open(morph_phon_file, 'r') as f:
        data = json.load(f)

    # Extract predictions from the 'predictions' key (cross-validation aggregated)
    predictions = data['predictions']['y_pred']
    true_labels = data['predictions']['y_true']

    # Handle string labels (multiclass) - create label map
    label_map = None
    if predictions and isinstance(predictions[0], str):
        unique_labels = sorted(set(predictions + true_labels))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        # Convert to integers
        predictions = [label_map[p] for p in predictions]
        true_labels = [label_map[t] for t in true_labels]

    return {
        'record_ids': data['predictions']['record_ids'],
        'predictions': predictions,
        'probabilities': data['predictions']['y_proba'],
        'true_labels': true_labels,
        'label_map': label_map
    }

def compute_confidence_categories(probabilities, predictions):
    """Categorize predictions by confidence level."""
    confidence_scores = []
    for prob, pred in zip(probabilities, predictions):
        # For multiclass, get probability of predicted class
        if isinstance(prob, list):
            confidence = prob[pred]
        else:
            # Binary classification
            confidence = prob if pred == 1 else (1 - prob)
        confidence_scores.append(confidence)

    categories = []
    for conf in confidence_scores:
        if conf >= 0.8:
            categories.append('High (≥0.8)')
        elif conf >= 0.6:
            categories.append('Medium (0.6-0.8)')
        else:
            categories.append('Low (<0.6)')

    return confidence_scores, categories

def analyze_error_overlap(lstm_data, morph_phon_data):
    """Analyze overlap between LSTM and Morph+Phon errors."""
    # Create record-level mapping
    lstm_map = {rid: (pred, true) for rid, pred, true in
                zip(lstm_data['record_ids'],
                    lstm_data['predictions'],
                    lstm_data['true_labels'])}

    mp_map = {rid: (pred, true) for rid, pred, true in
              zip(morph_phon_data['record_ids'],
                  morph_phon_data['predictions'],
                  morph_phon_data['true_labels'])}

    # Find common records (intersection)
    common_records = set(lstm_map.keys()) & set(mp_map.keys())

    # Categorize errors
    lstm_only_errors = []
    mp_only_errors = []
    both_errors = []
    both_correct = []

    for rid in common_records:
        lstm_pred, true_label = lstm_map[rid]
        mp_pred, _ = mp_map[rid]

        lstm_correct = (lstm_pred == true_label)
        mp_correct = (mp_pred == true_label)

        if not lstm_correct and not mp_correct:
            both_errors.append(rid)
        elif not lstm_correct and mp_correct:
            lstm_only_errors.append(rid)
        elif lstm_correct and not mp_correct:
            mp_only_errors.append(rid)
        else:
            both_correct.append(rid)

    return {
        'lstm_only_errors': lstm_only_errors,
        'mp_only_errors': mp_only_errors,
        'both_errors': both_errors,
        'both_correct': both_correct,
        'total_records': len(common_records)
    }

def rank_idiosyncrasies(lstm_data, morph_phon_data, overlap_analysis, top_n=20):
    """Rank forms by idiosyncrasy (hardest to predict)."""
    # Create confidence scores
    lstm_conf, _ = compute_confidence_categories(
        lstm_data['probabilities'],
        lstm_data['predictions']
    )
    mp_conf, _ = compute_confidence_categories(
        morph_phon_data['probabilities'],
        morph_phon_data['predictions']
    )

    # Create mapping
    lstm_conf_map = dict(zip(lstm_data['record_ids'], lstm_conf))
    mp_conf_map = dict(zip(morph_phon_data['record_ids'], mp_conf))

    # Score each record
    idiosyncrasy_scores = []

    for rid in overlap_analysis['lstm_only_errors'] + \
                overlap_analysis['mp_only_errors'] + \
                overlap_analysis['both_errors']:

        lstm_c = lstm_conf_map.get(rid, 0)
        mp_c = mp_conf_map.get(rid, 0)

        # Idiosyncrasy score: average confidence when wrong
        # Higher confidence but wrong = more idiosyncratic
        if rid in overlap_analysis['both_errors']:
            # Both models wrong despite confidence
            score = (lstm_c + mp_c) / 2
            error_type = 'Both Fail'
        elif rid in overlap_analysis['lstm_only_errors']:
            score = lstm_c
            error_type = 'LSTM Fail'
        else:
            score = mp_c
            error_type = 'MP Fail'

        idiosyncrasy_scores.append({
            'record_id': rid,
            'idiosyncrasy_score': score,
            'error_type': error_type,
            'lstm_confidence': lstm_c,
            'mp_confidence': mp_c
        })

    # Sort by idiosyncrasy score (descending)
    idiosyncrasy_scores.sort(key=lambda x: x['idiosyncrasy_score'], reverse=True)

    return idiosyncrasy_scores[:top_n]

def load_noun_data():
    """Load main dataset for noun information."""
    df = pd.read_csv('data/tash_nouns.csv')
    return df

def create_venn_diagram(overlap_analysis, domain, output_dir):
    """Create Venn diagram of error overlap."""
    plt.figure(figsize=(10, 8))

    # Set counts
    lstm_only = len(overlap_analysis['lstm_only_errors'])
    mp_only = len(overlap_analysis['mp_only_errors'])
    both = len(overlap_analysis['both_errors'])

    # Create Venn diagram
    venn = venn2(subsets=(lstm_only, mp_only, both),
                 set_labels=('LSTM Errors', 'Morph+Phon Errors'))

    # Customize colors
    if venn.get_patch_by_id('10'):
        venn.get_patch_by_id('10').set_color('#FFB6C1')  # Light red for LSTM-only
    if venn.get_patch_by_id('01'):
        venn.get_patch_by_id('01').set_color('#ADD8E6')  # Light blue for MP-only
    if venn.get_patch_by_id('11'):
        venn.get_patch_by_id('11').set_color('#DDA0DD')  # Plum for overlap

    # Add title
    total_errors = lstm_only + mp_only + both
    plt.title(f'Error Overlap Analysis: {domain.upper()}\n'
              f'Total Errors: {total_errors} / {overlap_analysis["total_records"]} '
              f'({100*total_errors/overlap_analysis["total_records"]:.1f}%)',
              fontsize=14, fontweight='bold')

    # Save
    output_path = output_dir / f'error_venn_{domain}.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Venn diagram saved to {output_path}")

def generate_reports(domain, lstm_data, morph_phon_data, overlap_analysis,
                    idiosyncrasy_ranking, output_dir):
    """Generate comprehensive reports."""

    # Load noun data for enrichment
    df = load_noun_data()

    # 1. Confidence Analysis Report
    lstm_conf, lstm_cat = compute_confidence_categories(
        lstm_data['probabilities'],
        lstm_data['predictions']
    )
    mp_conf, mp_cat = compute_confidence_categories(
        morph_phon_data['probabilities'],
        morph_phon_data['predictions']
    )

    confidence_df = pd.DataFrame({
        'record_id': lstm_data['record_ids'],
        'true_label': lstm_data['true_labels'],
        'lstm_prediction': lstm_data['predictions'],
        'lstm_confidence': lstm_conf,
        'lstm_category': lstm_cat,
        'mp_prediction': [morph_phon_data['predictions'][morph_phon_data['record_ids'].index(rid)]
                         if rid in morph_phon_data['record_ids'] else None
                         for rid in lstm_data['record_ids']],
        'mp_confidence': [mp_conf[morph_phon_data['record_ids'].index(rid)]
                         if rid in morph_phon_data['record_ids'] else None
                         for rid in lstm_data['record_ids']],
        'mp_category': [mp_cat[morph_phon_data['record_ids'].index(rid)]
                       if rid in morph_phon_data['record_ids'] else None
                       for rid in lstm_data['record_ids']]
    })

    confidence_output = output_dir / f'confidence_analysis_{domain}.csv'
    confidence_df.to_csv(confidence_output, index=False)
    print(f"Confidence analysis saved to {confidence_output}")

    # 2. Error Overlap Report
    overlap_df = pd.DataFrame({
        'Category': ['LSTM Only', 'Morph+Phon Only', 'Both Models Fail', 'Both Correct', 'Total'],
        'Count': [
            len(overlap_analysis['lstm_only_errors']),
            len(overlap_analysis['mp_only_errors']),
            len(overlap_analysis['both_errors']),
            len(overlap_analysis['both_correct']),
            overlap_analysis['total_records']
        ],
        'Percentage': [
            100 * len(overlap_analysis['lstm_only_errors']) / overlap_analysis['total_records'],
            100 * len(overlap_analysis['mp_only_errors']) / overlap_analysis['total_records'],
            100 * len(overlap_analysis['both_errors']) / overlap_analysis['total_records'],
            100 * len(overlap_analysis['both_correct']) / overlap_analysis['total_records'],
            100.0
        ]
    })

    overlap_output = output_dir / f'error_overlap_{domain}.csv'
    overlap_df.to_csv(overlap_output, index=False)
    print(f"Error overlap analysis saved to {overlap_output}")

    # 3. Idiosyncrasy Ranking Report (with noun details)
    idio_records = []
    for item in idiosyncrasy_ranking:
        rid = item['record_id']
        noun_row = df[df['recordID'] == rid]

        if not noun_row.empty:
            noun_row = noun_row.iloc[0]
            idio_records.append({
                'Rank': len(idio_records) + 1,
                'RecordID': rid,
                'SingularTheme': noun_row.get('analysisSingularTheme', ''),
                'PluralTheme': noun_row.get('analysisPluralTheme', ''),
                'InternalChanges': noun_row.get('analysisInternalChanges', ''),
                'GlossEnglish': noun_row.get('lexiconGlossEnglish', ''),
                'ErrorType': item['error_type'],
                'IdiosyncracyScore': f"{item['idiosyncrasy_score']:.3f}",
                'LSTM_Confidence': f"{item['lstm_confidence']:.3f}",
                'MP_Confidence': f"{item['mp_confidence']:.3f}"
            })

    idio_df = pd.DataFrame(idio_records)
    idio_output = output_dir / f'idiosyncrasy_top20_{domain}.csv'
    idio_df.to_csv(idio_output, index=False)
    print(f"Idiosyncrasy ranking saved to {idio_output}")

    # 4. Summary Statistics
    print("\n" + "="*60)
    print(f"RESIDUAL ANALYSIS SUMMARY: {domain.upper()}")
    print("="*60)
    print(f"\nTotal Records Analyzed: {overlap_analysis['total_records']}")
    print(f"\nError Breakdown:")
    print(f"  - LSTM-only errors: {len(overlap_analysis['lstm_only_errors'])} "
          f"({100*len(overlap_analysis['lstm_only_errors'])/overlap_analysis['total_records']:.1f}%)")
    print(f"  - Morph+Phon-only errors: {len(overlap_analysis['mp_only_errors'])} "
          f"({100*len(overlap_analysis['mp_only_errors'])/overlap_analysis['total_records']:.1f}%)")
    print(f"  - Both models fail: {len(overlap_analysis['both_errors'])} "
          f"({100*len(overlap_analysis['both_errors'])/overlap_analysis['total_records']:.1f}%)")
    print(f"  - Both models correct: {len(overlap_analysis['both_correct'])} "
          f"({100*len(overlap_analysis['both_correct'])/overlap_analysis['total_records']:.1f}%)")

    total_errors = (len(overlap_analysis['lstm_only_errors']) +
                   len(overlap_analysis['mp_only_errors']) +
                   len(overlap_analysis['both_errors']))
    print(f"\nTotal Error Rate: {100*total_errors/overlap_analysis['total_records']:.1f}%")

    if total_errors > 0:
        print(f"\nError Distribution:")
        print(f"  - LSTM-only: {100*len(overlap_analysis['lstm_only_errors'])/total_errors:.1f}%")
        print(f"  - Morph+Phon-only: {100*len(overlap_analysis['mp_only_errors'])/total_errors:.1f}%")
        print(f"  - Both fail: {100*len(overlap_analysis['both_errors'])/total_errors:.1f}%")

    print(f"\nTop 5 Most Idiosyncratic Forms:")
    for i, item in enumerate(idio_records[:5]):
        print(f"  {i+1}. {item['SingularTheme']} → {item['PluralTheme']} "
              f"(Score: {item['IdiosyncracyScore']}, {item['ErrorType']})")

    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description='Analyze LSTM residuals and idiosyncrasies')
    parser.add_argument('--domain', required=True,
                       choices=['has_suffix', 'has_mutation', '3way', 'ablaut', 'medial_a',
                              'final_a', 'final_vw', 'insert_c', 'templatic', '8way'],
                       help='Domain to analyze')
    parser.add_argument('--top-n', type=int, default=20,
                       help='Number of top idiosyncratic forms to report (default: 20)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path('reports') / 'lstm_residuals'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading LSTM baseline predictions for {args.domain}...")
    lstm_data = load_lstm_predictions(args.domain)

    print(f"Loading Morph+Phon ablation predictions for {args.domain}...")
    morph_phon_data = load_morph_phon_predictions(args.domain)

    print("Analyzing error overlap...")
    overlap_analysis = analyze_error_overlap(lstm_data, morph_phon_data)

    print(f"Ranking top {args.top_n} idiosyncratic forms...")
    idiosyncrasy_ranking = rank_idiosyncrasies(
        lstm_data, morph_phon_data, overlap_analysis, top_n=args.top_n
    )

    print("Generating Venn diagram...")
    create_venn_diagram(overlap_analysis, args.domain, output_dir)

    print("Generating comprehensive reports...")
    generate_reports(
        args.domain, lstm_data, morph_phon_data,
        overlap_analysis, idiosyncrasy_ranking, output_dir
    )

    print(f"\n✅ Residual analysis complete for {args.domain}")

if __name__ == '__main__':
    main()
