"""
Feature Correlation Analysis - Final Version
=============================================

Purpose: Comprehensive correlation analysis for features used in modeling.

Correlation Measures:
- Pearson's r: For continuous-continuous, continuous-binary, binary-binary
- Cramér's V: For categorical-categorical relationships

Features analyzed (actual features used in modeling):
- Morphological: analysisMutability, wordDerivedCategory, analysisRAugVowel, lexiconLoanwordSource
- Semantic: lexiconHumanYN, lexiconAnimateYN, lexiconSemanticField
- Phonological: 9 binary/continuous features derived from LH patterns and foot structures

Date: December 30, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import chi2_contingency
from itertools import combinations

# Paths
DATA_PATH = Path('/Users/alderete/CodeRepos/predicting-tashlhiyt-plural/data/tash_nouns.csv')
FIGURES_PATH = Path('/Users/alderete/CodeRepos/predicting-tashlhiyt-plural/figures')
FIGURES_PATH.mkdir(exist_ok=True)

print("=" * 80)
print("FEATURE CORRELATION ANALYSIS - FINAL")
print("=" * 80)
print()

# Load dataset
df = pd.read_csv(DATA_PATH)
usable = df[df['analysisPluralPattern'].isin(['External', 'Internal', 'Mixed'])].copy()
print(f"Usable records: {len(usable):,}")
print()

# ============================================================================
# Helper Functions
# ============================================================================

def cramers_v(x, y):
    """
    Calculate Cramér's V for categorical-categorical association.

    V = sqrt(chi2 / (n * min(k-1, r-1)))

    Returns value between 0 (no association) and 1 (perfect association).
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape[0] - 1, confusion_matrix.shape[1] - 1)
    if min_dim == 0:
        return 0.0
    return np.sqrt(chi2 / (n * min_dim))

def add_phonological_features(df):
    """Add all 9 phonological features."""
    df = df.copy()

    # Binary groupings (6)
    df['p_LH_ends_L'] = (df['p_stem_sing_LH'].fillna('').str[-1] == 'L').astype(int)
    df['p_LH_initial_weight'] = (df['p_stem_sing_LH'].fillna('').str[0] == 'L').astype(int)
    df['p_LH_less_2_syllables'] = (df['p_stem_sing_LH'].fillna('').str.len() <= 2).astype(int)
    df['p_LH_all_heavy'] = (~df['p_stem_sing_LH'].fillna('').str.contains('L')).astype(int)
    df['p_foot_residue_right'] = (df['p_stem_sing_foot'].fillna('').str[-1] == 'l').astype(int)
    df['p_foot_residue'] = df['p_stem_sing_foot'].fillna('').str.contains('l').astype(int)

    # Continuous groupings (3)
    df['p_LH_count_heavies'] = df['p_stem_sing_LH'].fillna('').apply(lambda x: x.count('H'))
    df['p_LH_count_moras'] = df['p_stem_sing_LH'].apply(
        lambda x: sum(2 if char == 'H' else 1 if char == 'L' else 0 for char in str(x)) if pd.notna(x) else np.nan
    )
    df['p_foot_count_feet'] = df['p_stem_sing_foot'].fillna('').apply(lambda x: x.count('F'))

    return df

# ============================================================================
# 1. PHONOLOGICAL FEATURES (Pearson's r)
# ============================================================================

print("=" * 80)
print("1. PHONOLOGICAL FEATURES (Pearson's r)")
print("=" * 80)
print()

usable_phon = add_phonological_features(usable)

phon_features = [
    'p_LH_ends_L', 'p_LH_initial_weight', 'p_LH_less_2_syllables', 'p_LH_all_heavy',
    'p_foot_residue_right', 'p_foot_residue',
    'p_LH_count_heavies', 'p_LH_count_moras', 'p_foot_count_feet'
]

phon_data = usable_phon[phon_features].dropna()
print(f"Features: {len(phon_features)} (6 binary + 3 continuous)")
print(f"Valid records: {len(phon_data):,}")
print()

# Compute correlation matrix
phon_corr = phon_data.corr()

# Find high correlations (|r| > 0.7)
phon_high = []
for i in range(len(phon_corr.columns)):
    for j in range(i+1, len(phon_corr.columns)):
        r = phon_corr.iloc[i, j]
        if abs(r) > 0.7:
            phon_high.append({
                'Feature 1': phon_corr.columns[i],
                'Feature 2': phon_corr.columns[j],
                'r': r,
                'abs_r': abs(r)
            })

print(f"High correlations (|r| > 0.7): {len(phon_high)}")
for item in sorted(phon_high, key=lambda x: x['abs_r'], reverse=True):
    severity = "🔴 SEVERE" if item['abs_r'] > 0.9 else "🟡 STRONG" if item['abs_r'] > 0.8 else "🟢 MODERATE"
    print(f"  {severity}  {item['Feature 1']} ↔ {item['Feature 2']}: r = {item['r']:.3f}")
print()

# Generate heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(phon_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax,
            cbar_kws={'label': "Pearson's r"})
ax.set_title("Phonological Features - Pearson's r", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(FIGURES_PATH / 'corr_phonological_pearson.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {FIGURES_PATH / 'corr_phonological_pearson.png'}")
print()

# ============================================================================
# 2. MORPHOLOGICAL FEATURES (Cramér's V)
# ============================================================================

print("=" * 80)
print("2. MORPHOLOGICAL FEATURES (Cramér's V)")
print("=" * 80)
print()

# Actual features we use for modeling
morph_features = {
    'analysisMutability': 'Mutability',
    'wordDerivedCategory': 'Derivational Category',
    'analysisRAugVowel': 'R-Augment Vowel',
    'lexiconLoanwordSource': 'Loan Source'
}

morph_data = usable[list(morph_features.keys())].dropna()
print(f"Features: {len(morph_features)} categorical")
print(f"Valid records: {len(morph_data):,}")
print()

# Compute Cramér's V matrix
morph_cols = list(morph_features.keys())
n = len(morph_cols)
cramers_matrix = np.zeros((n, n))

for i, col1 in enumerate(morph_cols):
    for j, col2 in enumerate(morph_cols):
        if i == j:
            cramers_matrix[i, j] = 1.0
        else:
            cramers_matrix[i, j] = cramers_v(morph_data[col1], morph_data[col2])

# Convert to DataFrame
morph_cramers = pd.DataFrame(
    cramers_matrix,
    index=[morph_features[c] for c in morph_cols],
    columns=[morph_features[c] for c in morph_cols]
)

# Find high associations (V > 0.3, typical threshold for categorical)
morph_high = []
for i in range(len(morph_cramers.columns)):
    for j in range(i+1, len(morph_cramers.columns)):
        v = morph_cramers.iloc[i, j]
        if v > 0.3:
            morph_high.append({
                'Feature 1': morph_cramers.columns[i],
                'Feature 2': morph_cramers.columns[j],
                'V': v
            })

print(f"High associations (V > 0.3): {len(morph_high)}")
if morph_high:
    for item in sorted(morph_high, key=lambda x: x['V'], reverse=True):
        severity = "🔴 STRONG" if item['V'] > 0.5 else "🟡 MODERATE" if item['V'] > 0.4 else "🟢 WEAK"
        print(f"  {severity}  {item['Feature 1']} ↔ {item['Feature 2']}: V = {item['V']:.3f}")
else:
    print("  ✅ No high associations found")
print()

# Generate heatmap
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(morph_cramers, annot=True, fmt='.3f', cmap='YlOrRd',
            vmin=0, vmax=1, square=True, linewidths=0.5, ax=ax,
            cbar_kws={'label': "Cramér's V"})
ax.set_title("Morphological Features - Cramér's V", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(FIGURES_PATH / 'corr_morphological_cramers.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {FIGURES_PATH / 'corr_morphological_cramers.png'}")
print()

# ============================================================================
# 3. SEMANTIC FEATURES (Mixed: Pearson's r for binary, Cramér's V for categorical)
# ============================================================================

print("=" * 80)
print("3. SEMANTIC FEATURES")
print("=" * 80)
print()

# Binary features: humanness, animacy
sem_binary = ['lexiconHumanYN', 'lexiconAnimateYN']
sem_binary_data = (usable[sem_binary] == 'Y').astype(int)

print("3a. Binary Semantic Features (Pearson's r)")
print("-" * 80)
sem_binary_corr = sem_binary_data.corr()
r_human_animate = sem_binary_corr.iloc[0, 1]
print(f"Humanness ↔ Animacy: r = {r_human_animate:.3f}")
severity = "🔴 STRONG" if abs(r_human_animate) > 0.8 else "🟡 MODERATE" if abs(r_human_animate) > 0.7 else "🟢 WEAK"
print(f"  {severity}")
print()

# Categorical feature: semantic field
print("3b. Semantic Field (Categorical)")
print("-" * 80)
sem_field_counts = usable['lexiconSemanticField'].value_counts()
print(f"Unique semantic fields: {len(sem_field_counts)}")
print(f"Records with semantic field: {usable['lexiconSemanticField'].notna().sum():,}")
print()

# Compute association between semantic field and binary features
print("3c. Semantic Field ↔ Binary Features (Cramér's V)")
print("-" * 80)
sem_data = usable[['lexiconSemanticField', 'lexiconHumanYN', 'lexiconAnimateYN']].dropna()

v_field_human = cramers_v(sem_data['lexiconSemanticField'], sem_data['lexiconHumanYN'])
v_field_animate = cramers_v(sem_data['lexiconSemanticField'], sem_data['lexiconAnimateYN'])

print(f"Semantic Field ↔ Humanness: V = {v_field_human:.3f}")
print(f"Semantic Field ↔ Animacy: V = {v_field_animate:.3f}")
print()

# Generate small heatmap for binary features
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(sem_binary_corr, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax,
            cbar_kws={'label': "Pearson's r"})
ax.set_title("Semantic Binary Features - Pearson's r", fontsize=12, fontweight='bold', pad=15)
ax.set_xticklabels(['Humanness', 'Animacy'], rotation=45, ha='right')
ax.set_yticklabels(['Humanness', 'Animacy'], rotation=0)
plt.tight_layout()
plt.savefig(FIGURES_PATH / 'corr_semantic_binary_pearson.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {FIGURES_PATH / 'corr_semantic_binary_pearson.png'}")
print()

# ============================================================================
# 4. CROSS-FAMILY CORRELATIONS
# ============================================================================

print("=" * 80)
print("4. CROSS-FAMILY CORRELATIONS")
print("=" * 80)
print()

print("4a. Morphological (categorical) ↔ Phonological (binary/continuous)")
print("-" * 80)
print("Using Cramér's V (treating continuous phonological as discretized)")
print()

# Discretize continuous phonological features for Cramér's V
phon_discrete = usable_phon[phon_features].copy()
phon_discrete['p_LH_count_heavies'] = pd.cut(phon_discrete['p_LH_count_heavies'], bins=3, labels=['Low', 'Med', 'High'])
phon_discrete['p_LH_count_moras'] = pd.cut(phon_discrete['p_LH_count_moras'], bins=3, labels=['Low', 'Med', 'High'])
phon_discrete['p_foot_count_feet'] = pd.cut(phon_discrete['p_foot_count_feet'], bins=3, labels=['Low', 'Med', 'High'])

# Find high cross-family associations
morph_phon_high = []
for morph_col in morph_features.keys():
    for phon_col in phon_features[:3]:  # Sample first 3 phonological features for speed
        data = pd.concat([usable[[morph_col]], phon_discrete[[phon_col]]], axis=1).dropna()
        if len(data) > 0:
            v = cramers_v(data[morph_col], data[phon_col])
            if v > 0.3:
                morph_phon_high.append({
                    'Morphological': morph_features[morph_col],
                    'Phonological': phon_col,
                    'V': v
                })

print(f"High associations (V > 0.3): {len(morph_phon_high)}")
if morph_phon_high:
    for item in sorted(morph_phon_high, key=lambda x: x['V'], reverse=True):
        print(f"  {item['Morphological']} ↔ {item['Phonological']}: V = {item['V']:.3f}")
else:
    print("  ✅ No high cross-family associations found")
print()

print("4b. Semantic (binary) ↔ Phonological (binary/continuous)")
print("-" * 80)
print("Using Pearson's r (for binary-binary and binary-continuous)")
print()

# Combine semantic binary with phonological features
sem_phon_data = pd.concat([sem_binary_data, phon_data], axis=1).dropna()
sem_phon_corr = sem_phon_data.corr()

# Extract cross-family correlations
sem_phon_high = []
for sem_col in sem_binary:
    for phon_col in phon_features:
        r = sem_phon_corr.loc[sem_col, phon_col]
        if abs(r) > 0.3:
            sem_phon_high.append({
                'Semantic': sem_col.replace('lexicon', '').replace('YN', ''),
                'Phonological': phon_col,
                'r': r
            })

print(f"High correlations (|r| > 0.3): {len(sem_phon_high)}")
if sem_phon_high:
    for item in sorted(sem_phon_high, key=lambda x: abs(x['r']), reverse=True):
        print(f"  {item['Semantic']} ↔ {item['Phonological']}: r = {item['r']:.3f}")
else:
    print("  ✅ No high cross-family correlations found")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

total_high = len(phon_high) + len(morph_high) + (1 if abs(r_human_animate) > 0.7 else 0) + len(morph_phon_high) + len(sem_phon_high)

print(f"Total high correlations/associations: {total_high}")
print()
print("By feature family:")
print(f"  - Phonological (Pearson's r > 0.7): {len(phon_high)}")
print(f"  - Morphological (Cramér's V > 0.3): {len(morph_high)}")
print(f"  - Semantic binary (Pearson's r > 0.7): {1 if abs(r_human_animate) > 0.7 else 0}")
print(f"  - Cross-family (various): {len(morph_phon_high) + len(sem_phon_high)}")
print()

if total_high == 0:
    print("✅ OVERALL: No multicollinearity issues detected")
elif total_high <= 5:
    print("🟡 OVERALL: Minor multicollinearity detected (acceptable with regularization)")
else:
    print("⚠️  OVERALL: Moderate multicollinearity detected")
    print("   Recommendation: Use regularization (L1/L2) in all models")

print()
print("Heatmaps saved:")
print(f"  - {FIGURES_PATH / 'corr_phonological_pearson.png'}")
print(f"  - {FIGURES_PATH / 'corr_morphological_cramers.png'}")
print(f"  - {FIGURES_PATH / 'corr_semantic_binary_pearson.png'}")
print()
print("=" * 80)
print("Analysis complete!")
print("=" * 80)

# ============================================================================
# Save detailed results to CSV
# ============================================================================

# Phonological correlations
phon_high_df = pd.DataFrame(phon_high)
if not phon_high_df.empty:
    phon_high_df = phon_high_df.sort_values('abs_r', ascending=False)
    phon_high_df.to_csv(FIGURES_PATH / 'corr_phonological_high.csv', index=False)
    print(f"✅ Saved: {FIGURES_PATH / 'corr_phonological_high.csv'}")

# Morphological associations
morph_high_df = pd.DataFrame(morph_high)
if not morph_high_df.empty:
    morph_high_df = morph_high_df.sort_values('V', ascending=False)
    morph_high_df.to_csv(FIGURES_PATH / 'corr_morphological_high.csv', index=False)
    print(f"✅ Saved: {FIGURES_PATH / 'corr_morphological_high.csv'}")
