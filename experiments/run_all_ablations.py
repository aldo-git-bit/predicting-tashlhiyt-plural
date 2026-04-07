"""
Master Ablation Runner - All 10 Domains

Runs ablation studies for all domains sequentially:
- Macro: has_suffix, has_mutation, 3way
- Micro: medial_a, final_a, final_vw, ablaut, insert_c, templatic, 8way

For each domain:
1. Builds ablation feature matrices (if not already built)
2. Runs ablation study (20 experiments)
3. Generates results table

Estimated total time: ~5-10 minutes (0.5-1 min per domain)

Usage:
    python experiments/run_all_ablations.py
    python experiments/run_all_ablations.py --skip-build  # Skip feature building
    python experiments/run_all_ablations.py --domains has_suffix medial_a  # Run specific domains only
"""

import argparse
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent

# All domains
ALL_DOMAINS = [
    # Macro domains
    'has_suffix',
    'has_mutation',
    '3way',
    # Micro domains
    'medial_a',
    'final_a',
    'final_vw',
    'ablaut',
    'insert_c',
    'templatic',
    '8way'
]


def check_features_exist(domain: str) -> bool:
    """Check if ablation features already exist for a domain."""
    features_dir = PROJECT_ROOT / 'features' / f'ablation_{domain}'
    return (features_dir.exists() and
            (features_dir / 'X_ngrams_only.csv').exists() and
            (features_dir / f'y_{domain}.csv').exists())


def build_features(domain: str):
    """Build ablation feature matrices for a domain."""
    print(f"\n{'='*80}")
    print(f"BUILDING FEATURES: {domain}")
    print(f"{'='*80}")

    cmd = ['python', 'scripts/build_ablation_features_all.py', '--domain', domain]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        print(f"\n✗ ERROR building features for {domain}")
        return False

    print(f"\n✓ Features built for {domain}")
    return True


def run_ablation(domain: str):
    """Run ablation study for a domain."""
    print(f"\n{'='*80}")
    print(f"RUNNING ABLATION: {domain}")
    print(f"{'='*80}")

    cmd = ['python', 'experiments/run_ablation.py', '--domain', domain]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        print(f"\n✗ ERROR running ablation for {domain}")
        return False

    print(f"\n✓ Ablation completed for {domain}")
    return True


def generate_table(domain: str):
    """Generate results table for a domain."""
    print(f"\n{'='*80}")
    print(f"GENERATING TABLE: {domain}")
    print(f"{'='*80}")

    cmd = ['python', 'experiments/generate_results_table.py', '--domain', domain]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        print(f"\n✗ ERROR generating table for {domain}")
        return False

    print(f"\n✓ Table generated for {domain}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Run ablation studies for all domains')
    parser.add_argument('--skip-build', action='store_true',
                       help='Skip feature building step (use existing features)')
    parser.add_argument('--skip-table', action='store_true',
                       help='Skip table generation step')
    parser.add_argument('--domains', nargs='+', default=ALL_DOMAINS,
                       help='Specific domains to run (default: all)')

    args = parser.parse_args()

    domains = args.domains
    skip_build = args.skip_build
    skip_table = args.skip_table

    print("="*80)
    print("MASTER ABLATION RUNNER")
    print("="*80)
    print(f"\nDomains to process ({len(domains)}):")
    for i, domain in enumerate(domains, 1):
        print(f"  {i}. {domain}")

    print(f"\nSteps:")
    print(f"  1. Build features: {'SKIP' if skip_build else 'YES'}")
    print(f"  2. Run ablation: YES")
    print(f"  3. Generate table: {'SKIP' if skip_table else 'YES'}")

    start_time = datetime.now()
    results = []

    for i, domain in enumerate(domains, 1):
        print(f"\n{'='*80}")
        print(f"DOMAIN {i}/{len(domains)}: {domain}")
        print(f"{'='*80}")

        domain_start = datetime.now()
        success = True

        # Step 1: Build features
        if not skip_build:
            if not check_features_exist(domain):
                if not build_features(domain):
                    success = False
            else:
                print(f"\n  ✓ Features already exist for {domain}, skipping build")

        # Step 2: Run ablation
        if success:
            if not run_ablation(domain):
                success = False

        # Step 3: Generate table
        if success and not skip_table:
            if not generate_table(domain):
                success = False

        domain_time = (datetime.now() - domain_start).total_seconds()

        results.append({
            'domain': domain,
            'success': success,
            'time': domain_time
        })

        print(f"\n{'='*80}")
        print(f"DOMAIN {domain}: {'✓ COMPLETE' if success else '✗ FAILED'}")
        print(f"Time: {domain_time/60:.1f} minutes")
        print(f"{'='*80}")

    total_time = (datetime.now() - start_time).total_seconds()

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Average per domain: {total_time/len(domains)/60:.1f} minutes")

    print(f"\nResults:")
    for r in results:
        status = "✓" if r['success'] else "✗"
        print(f"  {status} {r['domain']:15s} {r['time']/60:5.1f} min")

    successful = sum(1 for r in results if r['success'])
    print(f"\nSuccess rate: {successful}/{len(domains)} ({successful/len(domains)*100:.0f}%)")

    if successful == len(domains):
        print("\n🎉 All ablation studies completed successfully!")
    else:
        print(f"\n⚠️  {len(domains) - successful} domains failed")

    print(f"\n{'='*80}")


if __name__ == '__main__':
    main()
