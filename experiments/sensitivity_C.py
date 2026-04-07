"""
Sensitivity Analysis: Logistic Regression C Parameter
======================================================

Reviewer request: demonstrate that observed performance deltas between feature sets
are stable across C ∈ {0.1, 1.0, 10.0} and are not artifacts of the fixed C=1.0 choice.

Runs LR with each C value on all 6 feature sets × 10 domains.
Reports max Macro-F1 delta across C values per (domain, feature_set) cell
and confirms that feature-set rankings are preserved.

Output:
  reports/sensitivity_C_results.csv   – full F1 table (domain × feature_set × C)
  reports/sensitivity_C_summary.csv   – max delta per (domain, feature_set)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, run_cross_validation

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
ABLATION_DIR = PROJECT_ROOT / "features"
REPORTS_DIR  = PROJECT_ROOT / "reports"

C_VALUES = [0.1, 1.0, 10.0]

DOMAINS = [
    # (domain_name,  use_smote)
    ("has_suffix",   False),
    ("has_mutation", False),
    ("3way",         False),
    ("medial_a",     True),
    ("final_a",      True),
    ("final_vw",     True),
    ("ablaut",       False),
    ("insert_c",     True),
    ("templatic",    False),
    ("8way",         False),
]

FEATURE_SETS = [
    "ngrams_only",
    "semantic_only",
    "morph_only",
    "phon_only",
    "morph_phon",
    "all_features",
]

FEATURE_LABELS = {
    "ngrams_only":   "Ngr",
    "semantic_only": "Sem",
    "morph_only":    "Morph",
    "phon_only":     "Phon",
    "morph_phon":    "M+P",
    "all_features":  "All",
}


def load_xy(domain: str, feature_set: str):
    domain_dir = ABLATION_DIR / f"ablation_{domain}"
    X = pd.read_csv(domain_dir / f"X_{feature_set}.csv", index_col=0)
    y = pd.read_csv(domain_dir / f"y_{domain}.csv",      index_col=0).squeeze()
    return X, y


def run_lr(domain: str, feature_set: str, C: float,
           use_smote: bool, config: dict) -> float:
    """Return mean Macro-F1 for one (domain, feature_set, C) combination."""
    X, y = load_xy(domain, feature_set)

    # Encode string labels (needed for some multiclass domains)
    if y.dtype == object:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)

    model = LogisticRegression(
        C=C,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=config["random_state"],
        n_jobs=config["n_jobs"],
    )

    results = run_cross_validation(
        model, X, y,
        n_folds=config["cv"]["n_folds"],
        random_state=config["random_state"],
        use_smote=use_smote,
    )
    return results["overall_metrics"]["macro_f1_mean"]


def main():
    config = load_config()
    REPORTS_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("SENSITIVITY ANALYSIS: Logistic Regression C ∈ {0.1, 1.0, 10.0}")
    print(f"Domains: {len(DOMAINS)}  |  Feature sets: {len(FEATURE_SETS)}")
    print(f"Total runs: {len(DOMAINS) * len(FEATURE_SETS) * len(C_VALUES)}")
    print("=" * 70)

    rows = []
    total = len(DOMAINS) * len(FEATURE_SETS) * len(C_VALUES)
    done = 0
    t0 = datetime.now()

    for domain, use_smote in DOMAINS:
        print(f"\n── {domain} {'[SMOTE]' if use_smote else ''} ──")
        for fs in FEATURE_SETS:
            f1_by_c = {}
            for C in C_VALUES:
                f1 = run_lr(domain, fs, C, use_smote, config)
                f1_by_c[C] = f1
                done += 1
                elapsed = (datetime.now() - t0).total_seconds()
                eta = elapsed / done * (total - done)
                print(f"  {FEATURE_LABELS[fs]:6s}  C={C:4.1f}  F1={f1:.4f}"
                      f"  [{done}/{total}  ETA {eta/60:.1f} min]")

            rows.append({
                "domain":      domain,
                "feature_set": FEATURE_LABELS[fs],
                "C_0.1":       f1_by_c[0.1],
                "C_1.0":       f1_by_c[1.0],
                "C_10.0":      f1_by_c[10.0],
                "max_delta":   max(f1_by_c.values()) - min(f1_by_c.values()),
            })

    # ── Save full results ─────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    full_path = REPORTS_DIR / "sensitivity_C_results.csv"
    df.to_csv(full_path, index=False, float_format="%.4f")
    print(f"\nFull results saved → {full_path}")

    # ── Summary table: max delta per domain ──────────────────────────────────
    summary = (
        df.groupby("domain")["max_delta"]
          .agg(["max", "mean"])
          .rename(columns={"max": "worst_delta", "mean": "avg_delta"})
          .round(4)
    )
    summary_path = REPORTS_DIR / "sensitivity_C_summary.csv"
    summary.to_csv(summary_path, float_format="%.4f")
    print(f"Summary saved        → {summary_path}\n")

    # ── Print readable table ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS: Macro-F1 across C values (all feature sets × domains)")
    print("=" * 70)

    for domain, _ in DOMAINS:
        sub = df[df["domain"] == domain].copy()
        print(f"\n{domain}")
        print(f"  {'Feature':6s}  {'C=0.1':6s}  {'C=1.0':6s}  {'C=10':6s}  {'Δmax':5s}")
        print(f"  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*5}")
        for _, r in sub.iterrows():
            print(f"  {r['feature_set']:6s}  {r['C_0.1']:.4f}  {r['C_1.0']:.4f}"
                  f"  {r['C_10.0']:.4f}  {r['max_delta']:.4f}")

    # ── Overall verdict ───────────────────────────────────────────────────────
    overall_max   = df["max_delta"].max()
    overall_mean  = df["max_delta"].mean()
    pct_under_01  = (df["max_delta"] < 0.01).mean() * 100
    pct_under_02  = (df["max_delta"] < 0.02).mean() * 100

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  Largest single-cell Δ across C values : {overall_max:.4f}")
    print(f"  Mean Δ across all cells                : {overall_mean:.4f}")
    print(f"  Cells with Δ < 0.01                    : {pct_under_01:.1f}%")
    print(f"  Cells with Δ < 0.02                    : {pct_under_02:.1f}%")

    # Check whether M+P ranking vs N-grams is preserved across all C values
    ranking_preserved = True
    violations = []
    for domain, _ in DOMAINS:
        sub = df[df["domain"] == domain].set_index("feature_set")
        for C_col in ["C_0.1", "C_1.0", "C_10.0"]:
            if "M+P" in sub.index and "Ngr" in sub.index:
                if sub.loc["M+P", C_col] <= sub.loc["Ngr", C_col]:
                    ranking_preserved = False
                    violations.append((domain, C_col))

    if ranking_preserved:
        print("\n  ✓ M+P > N-grams ranking preserved across ALL C values and domains")
    else:
        print(f"\n  ✗ Ranking violations found: {violations}")

    total_time = (datetime.now() - t0).total_seconds()
    print(f"\nTotal runtime: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()
