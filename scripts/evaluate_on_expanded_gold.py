"""
Evaluate the updated syllabifier against the expanded golden dataset
"""
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

# Force reload
if 'rule_based_syllabifier' in sys.modules:
    del sys.modules['rule_based_syllabifier']

from rule_based_syllabifier import RuleBasedSyllabifier, SyllabificationEvaluator

# Load expanded golden dataset
gold_data = pd.read_csv(str(Path(__file__).parent.parent / 'data' / r'golden_syllables_expanded_simple.csv'))

# Filter out empty rows
gold_data = gold_data[gold_data['syllabifiedTheme'].notna() & (gold_data['syllabifiedTheme'] != '')]

# Prepare test data
test_data = [(row['analysisSingularTheme'], row['syllabifiedTheme'])
             for _, row in gold_data.iterrows()]

print(f"Total test cases: {len(test_data)}")
print()

# Evaluate
syllabifier = RuleBasedSyllabifier()
evaluator = SyllabificationEvaluator(syllabifier)
results = evaluator.evaluate(test_data)

# Print results
evaluator.print_evaluation(results)

# Group errors by pattern
if results['errors']:
    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)

    # Load full dataset with notes
    gold_full = pd.read_csv(str(Path(__file__).parent.parent / 'data' / r'golden_syllables_expanded.csv'))

    # Count errors by note category
    error_inputs = {e['input'] for e in results['errors']}
    error_rows = gold_full[gold_full['analysisSingularTheme'].isin(error_inputs)]

    if 'notes' in error_rows.columns:
        print("\nErrors by correction category:")
        note_counts = error_rows['notes'].value_counts()
        for note, count in note_counts.items():
            print(f"  {note:35s}: {count:3d} errors")

        print("\nErrors by source:")
        source_counts = error_rows['source'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source:20s}: {count:3d} errors")
