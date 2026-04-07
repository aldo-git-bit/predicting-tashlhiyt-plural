"""
Complete N-gram Feature Selection Workflow

Runs the entire pipeline from start to finish:
1. Feature selection for all 8 targets
2. Consolidation across targets
3. Report generation
4. Cross-validation
5. Save feature matrices to CSV

Usage:
    python run_complete_workflow.py [n_iterations]

Example:
    python run_complete_workflow.py 100
"""

import sys
from pathlib import Path
from datetime import datetime
import subprocess


def run_command(cmd, description):
    """Run a command and report status."""
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"\n❌ ERROR: {description} failed")
        return False

    print(f"\n✅ SUCCESS: {description}")
    return True


def main():
    n_iterations = 100
    if len(sys.argv) > 1:
        n_iterations = int(sys.argv[1])

    print(f"\n{'='*70}")
    print("N-GRAM FEATURE SELECTION - COMPLETE WORKFLOW")
    print(f"{'='*70}\n")
    print(f"Configuration:")
    print(f"  Bootstrap iterations: {n_iterations}")
    print(f"  Estimated time: {n_iterations * 8 * 2 / 60:.0f} minutes")
    print()

    # Step 1: Run feature selection
    if not run_command(
        f"python run_all_targets.py {n_iterations}",
        "Feature selection for all targets"
    ):
        return False

    # Find the timestamp of the most recent results
    results_base = Path("../../results/ngram_feature_selection")
    latest_dir = max(results_base.glob("*/"), key=lambda p: p.stat().st_mtime)
    timestamp = latest_dir.name

    print(f"\n📁 Results directory: {latest_dir}")

    # Step 2: Consolidate results
    if not run_command(
        f"python consolidate_features.py {latest_dir}",
        "Consolidate features across targets"
    ):
        return False

    # Step 3: Generate report
    consolidated_dir = latest_dir / "consolidated"
    if not run_command(
        f"python generate_report.py {consolidated_dir}",
        "Generate diagnostic report"
    ):
        return False

    # Step 4: Validate features
    if not run_command(
        f"python validate_features.py {timestamp}",
        "Cross-validation of selected features"
    ):
        return False

    # Step 5: Save feature matrices
    if not run_command(
        f"python save_feature_matrices.py {timestamp}",
        "Save feature matrices to CSV"
    ):
        return False

    # Final summary
    print(f"\n{'='*70}")
    print("WORKFLOW COMPLETE")
    print(f"{'='*70}\n")

    print(f"Results saved to: {latest_dir}")
    print(f"\nKey files:")
    print(f"  - Consolidated features: {consolidated_dir}")
    print(f"  - Report: {consolidated_dir}/feature_selection_report.md")
    print(f"  - Validation: {consolidated_dir}/validation_results.json")
    print(f"  - Feature matrices:")
    print(f"    • data/ngram_features_macro.csv (n=1,185)")
    print(f"    • data/ngram_features_micro.csv (n=562)")
    print()

    print("Next steps:")
    print("  1. Review the feature selection report:")
    print(f"     {consolidated_dir}/feature_selection_report.md")
    print("  2. Load feature matrices for modeling:")
    print("     df_macro = pd.read_csv('data/ngram_features_macro.csv')")
    print("     df_micro = pd.read_csv('data/ngram_features_micro.csv')")
    print("  3. Join with main dataset on 'recordID' column")
    print()

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
