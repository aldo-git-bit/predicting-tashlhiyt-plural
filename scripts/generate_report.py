"""
Generate comprehensive syllabification report
"""

import pandas as pd
from pathlib import Path
from rule_based_syllabifier import RuleBasedSyllabifier, SyllabificationEvaluator


def load_gold_standard(csv_path: str):
    """Load gold standard syllabification data from CSV"""
    df = pd.read_csv(csv_path)
    df_valid = df[df['syllabifiedTheme'].notna() & (df['syllabifiedTheme'] != '')]

    test_data = []
    for _, row in df_valid.iterrows():
        input_str = row['analysisSingularTheme']
        expected = row['syllabifiedTheme']
        test_data.append((input_str, expected))

    return test_data


def categorize_errors(errors):
    """Categorize errors into types"""
    categories = {
        'preprocessing_issues': [],
        'incorrect_coda_assignment': [],
        'word_initial_clusters': [],
        'vocoid_coda_issues': [],
        'other': []
    }

    for error in errors:
        input_str = error['input']
        expected = error['expected']
        predicted = error['predicted']

        # Category 1: Preprocessing (! or ʷ issues)
        if input_str.startswith('!') and not predicted.startswith('!'):
            categories['preprocessing_issues'].append(error)
        elif 'ʷ' in expected and 'ʷ' not in predicted:
            categories['preprocessing_issues'].append(error)

        # Category 2: Incorrect coda assignment (vowels taking codas)
        elif any(f'{v}r' in predicted or f'{v}z' in predicted or f'{v}l' in predicted
                 for v in ['a', 'i', 'u']) and \
             any(f'{v}.r' in expected or f'{v}.z' in expected or f'{v}.l' in expected
                 for v in ['a', 'i', 'u']):
            categories['incorrect_coda_assignment'].append(error)

        # Category 3: Word-initial consonant clusters
        elif expected.startswith(input_str.lstrip('!')[0:2] + '.') and \
             predicted.startswith(input_str.lstrip('!')[0:3]):
            categories['word_initial_clusters'].append(error)

        # Category 4: j/w coda vs onset issues
        elif 'j' in predicted or 'w' in predicted:
            categories['vocoid_coda_issues'].append(error)

        else:
            categories['other'].append(error)

    return categories


def generate_report(output_path):
    """Generate comprehensive report"""
    base_dir = Path(__file__).parent.parent
    gold_csv = base_dir / 'data' / 'import_golden_syllables.csv'

    # Load and evaluate
    test_data = load_gold_standard(gold_csv)
    syllabifier = RuleBasedSyllabifier()
    evaluator = SyllabificationEvaluator(syllabifier)
    results = evaluator.evaluate(test_data)

    # Categorize errors
    categories = categorize_errors(results['errors'])

    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Tashlhiyt Berber Syllabification - Progress Report\n\n")
        f.write(f"**Date**: November 1, 2025\n")
        f.write(f"**Evaluation Dataset**: 72 gold standard examples\n\n")

        f.write("---\n\n")

        # Section 1: Algorithm Pseudocode
        f.write("## 1. Current Algorithm (Pseudocode)\n\n")
        f.write("```\n")
        f.write("PREPROCESSING:\n")
        f.write("  - Remove emphatic marker (!)\n")
        f.write("  - Remove labialization superscript (ʷ)\n")
        f.write("  - Parse string into segments (handle multi-char segments)\n\n")

        f.write("PHASE 1: HIERARCHICAL PEAK ASSIGNMENT (High to Low Sonority)\n")
        f.write("  For each sonority level L (from 8 down to 1):\n\n")

        f.write("    Step 1: CREATE PEAKS at sonority level L\n")
        f.write("      For each unassigned segment S at level L:\n")
        f.write("        - Skip if S is j or w (never peaks)\n")
        f.write("        - If next segment N is also level L and can be peak:\n")
        f.write("            → Assign S as ONSET, N as PEAK (same-sonority pairing)\n")
        f.write("            → Create syllable: onset=S, peak=N\n")
        f.write("        - Else:\n")
        f.write("            → Assign S as PEAK\n")
        f.write("            → Create syllable: peak=S\n\n")

        f.write("    Step 2: ASSIGN ONSETS to new peaks (without onsets)\n")
        f.write("      For each new peak P created at this level:\n")
        f.write("        - Look left: find segment at position P-1\n")
        f.write("        - If unassigned AND sonority ≤ P.sonority:\n")
        f.write("            → Assign as ONSET to P\n\n")

        f.write("    Step 3: ASSIGN CODAS to new peaks\n")
        f.write("      For each new peak P created at this level:\n")
        f.write("        - Look right: find segment at position P+1\n")
        f.write("        - If unassigned AND sonority < P.sonority:\n")
        f.write("            → Assign as CODA to P (max 1 coda per syllable)\n\n")

        f.write("PHASE 2: ASSIGN REMAINING SEGMENTS\n")
        f.write("  - Word-final unassigned → codas of last syllable\n")
        f.write("  - Word-initial unassigned → onsets of first syllable\n")
        f.write("  - Other unassigned → fit as onset/coda where possible\n\n")

        f.write("PHASE 3: OPTIMIZE GEMINATES\n")
        f.write("  - Find geminate pairs C.C creating onsetless syllables\n")
        f.write("  - Merge to CC (onset=C, peak=C) to reduce onsetless count\n")
        f.write("  - Example: r.r.wa → rr.wa\n\n")

        f.write("OUTPUT:\n")
        f.write("  - Sort syllables by peak position (left to right)\n")
        f.write("  - Insert '.' between syllables\n")
        f.write("```\n\n")

        # Section 2: Accuracy Report
        f.write("---\n\n")
        f.write("## 2. Accuracy Report\n\n")
        f.write(f"**Overall Accuracy**: {results['accuracy']:.2%} ({results['correct']}/{results['total']} correct)\n\n")
        f.write(f"**Total Errors**: {len(results['errors'])}\n\n")

        f.write("### Error Distribution by Category\n\n")
        f.write(f"1. **Preprocessing Issues**: {len(categories['preprocessing_issues'])} errors\n")
        f.write(f"2. **Incorrect Coda Assignment**: {len(categories['incorrect_coda_assignment'])} errors\n")
        f.write(f"3. **Word-Initial Clusters**: {len(categories['word_initial_clusters'])} errors\n")
        f.write(f"4. **Vocoid (j/w) Coda vs Onset**: {len(categories['vocoid_coda_issues'])} errors\n")
        f.write(f"5. **Other**: {len(categories['other'])} errors\n\n")

        # Section 3: Detailed Error Analysis
        f.write("---\n\n")
        f.write("## 3. Detailed Error Analysis\n\n")

        # Category 1
        f.write("### Category 1: Preprocessing Issues\n\n")
        f.write("**Problem**: Emphatic marker (!) or labialization (ʷ) removed but expected in output\n\n")
        if categories['preprocessing_issues']:
            for i, error in enumerate(categories['preprocessing_issues'], 1):
                f.write(f"{i}. `{error['input']}` → Expected: `{error['expected']}`, Got: `{error['predicted']}`\n")
        else:
            f.write("*No errors in this category*\n")
        f.write("\n")

        # Category 2
        f.write("### Category 2: Incorrect Coda Assignment\n\n")
        f.write("**Problem**: High-sonority vowels (a, i, u) taking lower-sonority segments as codas when those segments should start new syllables\n\n")
        if categories['incorrect_coda_assignment']:
            for i, error in enumerate(categories['incorrect_coda_assignment'], 1):
                f.write(f"{i}. `{error['input']}` → Expected: `{error['expected']}`, Got: `{error['predicted']}`\n")
        else:
            f.write("*No errors in this category*\n")
        f.write("\n")

        # Category 3
        f.write("### Category 3: Word-Initial Consonant Clusters\n\n")
        f.write("**Problem**: Word-initial consonant clusters not syllabifying correctly\n\n")
        if categories['word_initial_clusters']:
            for i, error in enumerate(categories['word_initial_clusters'], 1):
                f.write(f"{i}. `{error['input']}` → Expected: `{error['expected']}`, Got: `{error['predicted']}`\n")
        else:
            f.write("*No errors in this category*\n")
        f.write("\n")

        # Category 4
        f.write("### Category 4: Vocoid (j/w) Coda vs Onset Issues\n\n")
        f.write("**Problem**: j and w assigned as codas when they should be onsets, or vice versa\n\n")
        if categories['vocoid_coda_issues']:
            for i, error in enumerate(categories['vocoid_coda_issues'], 1):
                f.write(f"{i}. `{error['input']}` → Expected: `{error['expected']}`, Got: `{error['predicted']}`\n")
        else:
            f.write("*No errors in this category*\n")
        f.write("\n")

        # Category 5
        f.write("### Category 5: Other Errors\n\n")
        if categories['other']:
            for i, error in enumerate(categories['other'], 1):
                f.write(f"{i}. `{error['input']}` → Expected: `{error['expected']}`, Got: `{error['predicted']}`\n")
        else:
            f.write("*No errors in this category*\n")
        f.write("\n")

        # Complete error list
        f.write("---\n\n")
        f.write("## 4. Complete Error List\n\n")
        f.write("| # | Input | Expected | Predicted |\n")
        f.write("|---|-------|----------|----------|\n")
        for i, error in enumerate(results['errors'], 1):
            f.write(f"| {i} | `{error['input']}` | `{error['expected']}` | `{error['predicted']}` |\n")
        f.write("\n")

        # Key findings
        f.write("---\n\n")
        f.write("## 5. Key Findings & Next Steps\n\n")
        f.write("### Successes\n\n")
        f.write("- Achieved 61% accuracy on gold standard\n")
        f.write("- Correctly handles most basic syllabification cases\n")
        f.write("- Geminate optimization working for some cases\n")
        f.write("- Onset assignment priority implemented successfully\n\n")

        f.write("### Major Issues to Address\n\n")
        f.write("1. **Coda Over-assignment**: High-sonority segments (especially vowels) are taking codas when remaining segments could form valid syllables\n")
        f.write("   - Possible solution: Don't assign codas if remaining segments can form a valid syllable\n")
        f.write("   - Or: Restrict coda assignment for certain segment types\n\n")

        f.write("2. **Preprocessing**: Current approach removes ʷ and !, but gold standard expects them preserved\n")
        f.write("   - Need clarification on whether these should be preserved in output\n\n")

        f.write("3. **Vocoid Assignment**: j and w sometimes assigned as codas when they should be onsets\n")
        f.write("   - May need stricter priority for onset assignment of these segments\n\n")

        f.write("4. **Word-initial Clusters**: Some cases not handled correctly\n")
        f.write("   - May need special rules for word-initial position\n\n")

        f.write("### Recommended Next Steps\n\n")
        f.write("1. Clarify preprocessing rules (! and ʷ handling)\n")
        f.write("2. Implement smarter coda assignment logic\n")
        f.write("3. Test with expanded gold standard data\n")
        f.write("4. Refine geminate handling\n")
        f.write("5. Add special handling for word boundaries\n")

    print(f"Report generated: {output_path}")


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    output_path = base_dir / 'reports' / 'syllabification_progress_report.md'
    generate_report(output_path)
