# Predicting Tashlhiyt Plural Formation

A computational study of Tashlhiyt Berber (Tachelhit) nominal plural formation, combining rule-based phonological analysis with machine learning to quantify the predictability of plural patterns and identify lexically idiosyncratic forms.

---

## Overview

Tashlhiyt Berber nouns form plurals through three strategies: **external** (suffixation only), **internal** (stem mutation only), or **mixed** (both). This project investigates how much of plural formation is predictable from surface phonological form versus stored lexically, using a dataset of 1,914 nouns with full inflectional paradigms.

The central finding is that hand-crafted morphophonological features (syllable structure, foot type, morphological class) consistently outperform n-gram baselines and character-level neural models across 10 classification tasks, with Macro-F1 ranging from 0.59 (Final A insertion) to 0.91 (templatic mutation). Error overlap analysis reveals that only 19–25% of errors are made by both model types simultaneously, suggesting most failures reflect genuine lexical idiosyncrasy rather than underfitting.

---

## Repository Structure

```
predicting-tashlhiyt-plural/
│
├── data/
│   ├── tash_nouns.csv                  # Main dataset (1,914 nouns, 53 columns)
│   ├── tash_nouns_readme.txt           # Full column descriptions and documentation
│   ├── import_golden_syllables.csv     # 72-form gold standard for syllabification
│   ├── golden_syllables_expanded.csv   # Expanded gold standard (140 forms)
│   ├── forms_from_plural_theme.csv     # Plural forms derived from themes
│   ├── record_extractor_template.txt   # Template for extracting records by pattern
│   ├── ngram_features_macro.csv        # N-gram feature matrix, macro level (n=1,185)
│   ├── ngram_features_micro.csv        # N-gram feature matrix, micro level (n=562)
│   ├── ngram_metadata_macro.json       # N-gram feature selection metadata (macro)
│   ├── ngram_metadata_micro.json       # N-gram feature selection metadata (micro)
│   └── gold_standard/                  # Annotator validation files
│       ├── syllabified_stems_for_review_corrections.csv
│       ├── tash_nouns_check_LH_withCorrections.xlsx
│       ├── tash_nouns_check_LH2_corrected.xlsx
│       └── noninitial_geminates_11corrections.xlsx
│
├── scripts/                            # Data pipeline and analysis scripts
│   ├── rule_based_syllabifier.py       # Core syllabifier (99.5% accuracy)
│   ├── add_lh_column.py                # Light/Heavy pattern extraction
│   ├── add_foot_column.py              # Foot structure extraction
│   ├── feature_engineering.py          # Full feature engineering pipeline
│   ├── build_ablation_features_all.py  # Build feature matrices for ablation study
│   ├── build_lstm_data.py              # Preprocess data for LSTM experiments
│   ├── analyze_lstm_residuals.py       # Error overlap and idiosyncrasy analysis
│   ├── compute_significance_tests.py   # Paired t-tests across 10-fold CV results
│   ├── generate_master_tables.py       # Generate publication tables
│   ├── generate_appendix_tables.py     # Generate appendix tables (A1–A10)
│   ├── test_phonological_groupings.py  # Hypothesis-driven phonological analysis
│   └── [other scripts]                 # See scripts/ for full list
│
├── experiments/                        # ML experiment runners
│   ├── run_ablation.py                 # Single-domain ablation runner
│   ├── run_all_ablations.py            # Batch ablation runner (all 10 domains)
│   ├── rerun_morph_phon_logreg.py      # Re-run Morph+Phon LogReg with full outputs
│   ├── train_lstm.py                   # Bi-LSTM training pipeline
│   ├── sensitivity_C.py               # Logistic regression C sensitivity analysis
│   ├── utils.py                        # Shared CV and metric utilities
│   ├── config.yaml                     # Fixed hyperparameter configuration
│   └── results/                        # Experiment results (latest runs only)
│       ├── ablation_{domain}/          # Ablation summary CSV per domain
│       └── lstm_baseline_{domain}/     # LSTM summary CSV per domain
│
├── features/                           # Precomputed feature matrices
│   ├── X_macro_{task}.csv              # Feature matrix, macro tasks (n=1,185)
│   ├── X_micro_{task}.csv              # Feature matrix, micro tasks (n=562)
│   ├── y_macro_{task}.csv              # Labels, macro tasks
│   ├── y_micro_{task}.csv              # Labels, micro tasks
│   ├── lstm_data_{domain}.npz          # LSTM-ready encoded sequences
│   ├── char_vocab.json                 # Character vocabulary (28 characters)
│   ├── feature_metadata_{domain}.json  # Feature metadata per task
│   └── tashplur_variables.csv          # Variable definitions and descriptions
│
├── reports/
│   ├── master_table3_merged.csv        # Main results table (all 10 domains)
│   ├── master_table1_performance.csv   # Feature-set performance comparison
│   ├── master_table2_residual.csv      # Residual/idiosyncrasy analysis
│   ├── master_table2_with_significance.csv
│   ├── master_table3_with_significance.csv
│   ├── appendix_table_a11_significance.csv
│   ├── table2_corrected_latex.txt      # LaTeX for main table
│   ├── appendix_table_a11_latex_CORRECTED.txt
│   ├── sensitivity_C_results.csv       # C sensitivity full results
│   ├── sensitivity_C_summary.csv       # C sensitivity summary
│   ├── significance_tests_complete.csv
│   ├── phonological_groupings_comprehensive_results.json
│   ├── lh_to_foot_mapping_20251205.xlsx
│   ├── appendix_tables/                # Per-domain ablation tables (A1–A10)
│   └── lstm_residuals/                 # Error overlap, confidence, idiosyncrasy rankings
│
├── dashboard/
│   ├── app.py                          # Streamlit interactive dashboard
│   └── README.md                       # Dashboard documentation
│
├── figures/                            # Correlation and analysis figures
│
├── notes/
│   ├── LH_patterns.csv                 # Light/Heavy pattern reference
│   ├── foot_structures.csv             # Foot structure reference
│   ├── consonant_chart.png             # Tashlhiyt consonant inventory
│   ├── hypotheses_prosody.txt          # Prosodic hypotheses
│   └── grammar_synopsis.pdf            # Grammar reference
│
├── requirements.txt
└── .gitignore
```

---

## Dataset

**`data/tash_nouns.csv`** — 1,914 Tashlhiyt Berber nouns with:
- Full inflectional paradigm (8 forms: MSF, MSB, MPF, MPB, FSF, FSB, FPF, FPB)
- Morphological annotations (gender, mutability, derivational category, R-augment)
- Plural pattern classification (External / Internal / Mixed / No Plural)
- Internal change types (Ablaut, Templatic, Medial A, Final A, Final V/W, Insert C, Suppletion)
- Semantic annotations (animacy, humanness, semantic field, loanword source)
- Engineered phonological features (syllabification, Light/Heavy patterns, foot structure)

See **`data/tash_nouns_readme.txt`** for complete column descriptions and ML feature name mappings.

**Subset sizes for modeling:**
- Macro level (has_suffix, has_mutation, 3-way): n = 1,185
- Micro level (per mutation type, binary + 8-way): n = 562

---

## Classification Tasks

### Macro-Level (n = 1,185)
| Task | Description |
|------|-------------|
| `has_suffix` | Predicts presence of plural suffix (binary) |
| `has_mutation` | Predicts presence of stem mutation (binary) |
| `3way` | 3-way classification: External / Internal / Mixed |

### Micro-Level (n = 562, nouns with Internal or Mixed plural)
| Task | Description | Class imbalance |
|------|-------------|-----------------|
| `ablaut` | Vowel change (ablaut) | 48.0% minority |
| `templatic` | Root-and-pattern (templatic) | 18.1% minority |
| `medial_a` | Medial /a/ insertion | 16.5% minority |
| `final_a` | Final /a/ insertion | 12.6% minority |
| `insert_c` | Consonant insertion | 6.8% minority → SMOTE |
| `final_vw` | Final /v/ or /w/ insertion | 4.8% minority → SMOTE |
| `8way` | 8-way classification (6 mutations + 2 combinations) | — |

---

## Key Results (Macro-F1)

| Domain | N-grams | Morph+Phon | Bi-LSTM | Best |
|--------|---------|------------|---------|------|
| has_suffix | 0.849 | **0.876** | 0.753 | M+P |
| has_mutation | 0.739 | **0.769** | 0.644 | M+P |
| 3way | 0.572 | **0.683** | 0.531 | M+P |
| ablaut | 0.736 | **0.761** | 0.741 | M+P |
| templatic | 0.850 | **0.907** | 0.795 | M+P |
| medial_a | 0.718 | 0.715 | **0.673** | Ngr |
| final_a | 0.533 | **0.593** | 0.489 | M+P |
| insert_c | 0.636 | **0.636** | 0.480 | M+P |
| final_vw | 0.658 | 0.645 | **0.491** | Ngr |
| 8way | 0.357 | **0.469** | 0.373 | M+P |

Morph+Phon (LogReg) significantly outperforms bi-LSTM in 7/10 domains (paired t-test, p < 0.05). See `reports/master_table3_merged.csv` for complete results including residual analysis.

---

## Installation

```bash
# Clone the repo
git clone https://github.com/aldo-git-bit/predicting-tashlhiyt-plural.git
cd predicting-tashlhiyt-plural

# Install with uv (recommended)
pip install uv
uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt
```

**Python**: 3.13+ recommended. TensorFlow (for LSTM experiments) requires separate installation:
```bash
uv pip install tensorflow==2.15.0
```
See `experiments/train_lstm.py` for LSTM usage.

---

## Running the Pipeline

### 1. Regenerate phonological features (if starting from raw data)
```bash
python scripts/rule_based_syllabifier.py    # Verify syllabifier
python scripts/add_lh_column.py             # Light/Heavy patterns
python scripts/add_foot_column.py           # Foot structure
```

### 2. Build feature matrices for a domain
```bash
python scripts/build_ablation_features_all.py --domain has_suffix
python scripts/build_ablation_features_all.py --domain ablaut
# etc.
```

### 3. Run ablation experiments
```bash
# Single domain
python experiments/run_ablation.py --domain has_suffix

# All 10 domains (batch)
python experiments/run_all_ablations.py

# With SMOTE for imbalanced domains
python experiments/run_ablation.py --domain final_vw --use-smote
```

### 4. Train Bi-LSTM baseline
```bash
python scripts/build_lstm_data.py --domain has_suffix
python experiments/train_lstm.py --domain has_suffix
```

### 5. Generate publication tables
```bash
python scripts/generate_master_tables.py
python scripts/generate_appendix_tables.py
```

### 6. Run the dashboard
```bash
streamlit run dashboard/app.py
```

---

## Experimental Design

- **Cross-validation**: 10-fold stratified CV (random_state=42)
- **Hyperparameters**: Fixed across all domains (no tuning) — Logistic Regression C=1.0, Random Forest max_depth=10, XGBoost n_estimators=100/lr=0.1
- **Class balancing**: `class_weight='balanced'` for all models; SMOTE additionally applied to domains with extreme imbalance (final_vw, insert_c, final_a, medial_a)
- **Primary metric**: Macro-F1 (equal weight to both classes, critical for imbalanced tasks)
- **Significance testing**: Paired t-tests on 10-fold F1 scores; see `reports/significance_tests_complete.csv`
- **C sensitivity**: See `experiments/sensitivity_C.py` and `reports/sensitivity_C_results.csv` for robustness check across C ∈ {0.1, 1.0, 10.0}

---

## Citation

*Citation will be added upon publication.*

---

## Data Sources

The noun dataset was compiled from:
- Jebbour, A. (1996). *Phonologie et morphologie du tachelhit de Tiznit (Maroc)*. Ph.D. thesis, Université Paris V.
- Ouakrim, O. (1995). *Fonologie et morphologie du berbère*. Publications de la Faculté des Lettres et Sciences Humaines, Kénitra.

---

## License

Dataset and code are released for research use. See `data/tash_nouns_readme.txt` for full data provenance.
