"""
Evaluate the rule-based Tashlhiyt syllabifier against gold standard data.
"""

import pandas as pd
from pathlib import Path
from rule_based_syllabifier import RuleBasedSyllabifier, SyllabificationEvaluator


def load_gold_standard(csv_path: str):
    """Load gold standard syllabification data from CSV"""
    df = pd.read_csv(csv_path)

    # Filter out rows where syllabifiedTheme is empty
    df_valid = df[df['syllabifiedTheme'].notna() & (df['syllabifiedTheme'] != '')]

    # Create test data tuples
    test_data = []
    for _, row in df_valid.iterrows():
        input_str = row['analysisSingularTheme']
        expected = row['syllabifiedTheme']
        test_data.append((input_str, expected))

    print(f"Loaded {len(test_data)} gold standard examples")

    return test_data, df_valid


def analyze_errors(errors, syllabifier):
    """Analyze error patterns in detail"""
    if not errors:
        print("\nNo errors to analyze!")
        return

    print(f"\n{'='*80}")
    print(f"DETAILED ERROR ANALYSIS")
    print(f"{'='*80}\n")

    for i, error in enumerate(errors[:15], 1):  # Show first 15
        input_str = error['input']
        expected = error['expected']
        predicted = error['predicted']

        # Get debug info
        _, debug = syllabifier.syllabify(input_str, return_debug=True)

        print(f"Error #{i}: {input_str}")
        print(f"  Expected:  {expected}")
        print(f"  Predicted: {predicted}")
        print(f"  Segments:  {debug['segments']}")
        print(f"  Syllables: {debug['syllables']}")
        print()

    if len(errors) > 15:
        print(f"... and {len(errors) - 15} more errors\n")


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    gold_csv = base_dir / 'data' / 'import_golden_syllables.csv'

    # Load gold standard
    test_data, df_valid = load_gold_standard(gold_csv)

    if len(test_data) == 0:
        print("No valid test data found!")
        return

    # Initialize syllabifier
    syllabifier = RuleBasedSyllabifier()
    evaluator = SyllabificationEvaluator(syllabifier)

    # Run evaluation
    print(f"\n{'='*80}")
    print("Running evaluation...")
    print(f"{'='*80}\n")

    results = evaluator.evaluate(test_data)

    # Print results
    evaluator.print_evaluation(results)

    # Detailed error analysis
    if results['errors']:
        analyze_errors(results['errors'], syllabifier)


if __name__ == "__main__":
    main()
