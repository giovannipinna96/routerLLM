#!/usr/bin/env python3
"""
Debug script to understand validation issues
"""

import sys
import os
sys.path.append('src')

from datasets import load_dataset
import json

def debug_validation_data():
    """Debug the HumanEval Plus dataset to understand test cases"""
    print("Loading HumanEval Plus dataset for validation analysis...")

    try:
        # Load dataset
        dataset = load_dataset("evalplus/humanevalplus")
        examples = list(dataset["test"])

        print(f"Dataset loaded successfully. Total examples: {len(examples)}")

        # Find the problematic by_length example
        by_length_example = None
        for example in examples:
            if "by_length" in example.get('prompt', ''):
                by_length_example = example
                break

        if by_length_example:
            print("\n" + "="*60)
            print("BY_LENGTH EXAMPLE ANALYSIS")
            print("="*60)
            print(f"Task ID: {by_length_example.get('task_id', 'N/A')}")
            print(f"Entry point: {by_length_example.get('entry_point', 'N/A')}")
            print("\nPrompt:")
            print("-" * 30)
            print(by_length_example.get('prompt', 'NO PROMPT'))
            print("\nTest cases:")
            print("-" * 30)
            test_content = by_length_example.get('test', '')
            if test_content:
                print(test_content)
            else:
                print("NO TEST CASES FOUND!")

            print("\nPlus test cases:")
            print("-" * 30)
            plus_test_content = by_length_example.get('plus_test', '')
            if plus_test_content:
                print(plus_test_content)
            else:
                print("NO PLUS TEST CASES FOUND!")
        else:
            print("by_length example not found!")

        # Also check a few other examples
        print("\n" + "="*60)
        print("SAMPLE OF OTHER EXAMPLES")
        print("="*60)

        for i, example in enumerate(examples[:3]):
            print(f"\nExample {i+1}: {example.get('task_id', 'N/A')}")
            print(f"Has test: {'YES' if example.get('test') else 'NO'}")
            print(f"Has plus_test: {'YES' if example.get('plus_test') else 'NO'}")
            if example.get('test'):
                print(f"Test preview: {example['test'][:100]}...")

        return True

    except Exception as e:
        print(f"Failed to debug validation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_problematic_code():
    """Test the problematic by_length code manually"""
    print("\n" + "="*60)
    print("TESTING PROBLEMATIC CODE")
    print("="*60)

    # The actual code that was generated and marked as correct
    problematic_code = '''
def by_length(arr):
    if len(arr) == 0:
        return []
    else:
        sorted_arr = sorted(arr)
        reversed_arr = sorted_arr[::-1]
        result = []
        for num in reversed_arr:
            if num >= 1 and num <= 9:
                result.append("One" if num == 1 else "Two" if num == 2 else "Three" if num == 3 else "Four" if num == 4 else "Five" if num == 5 else "Six" if num == 6 else "Seven" if num == 7 else "Eight" if num == 8 else "Nine")
            elif num > 9:
                result.append('')
        return result
    '''

    print("Problematic code:")
    print(problematic_code)

    # Test with example from the problem description
    test_cases = [
        ([2, 1, 1, 4, 5, 8, 2, 3], ["Eight", "Five", "Four", "Three", "Two", "Two", "One", "One"]),
        ([], []),
        ([1, -1, 55], ['One']),
    ]

    print("\nTesting with manual test cases:")

    exec(problematic_code)  # Define the function

    for i, (input_arr, expected) in enumerate(test_cases):
        try:
            result = by_length(input_arr)
            passed = result == expected
            print(f"\nTest {i+1}: {input_arr}")
            print(f"Expected: {expected}")
            print(f"Got:      {result}")
            print(f"PASS: {passed}")
            if not passed:
                print("❌ FAILED!")
        except Exception as e:
            print(f"Test {i+1} ERROR: {e}")

def main():
    """Main debug function"""
    print("Starting Validation Debug Analysis")

    # First, analyze the dataset
    debug_validation_data()

    # Then, test the problematic code manually
    test_problematic_code()

    print("\nDebug analysis completed!")

if __name__ == "__main__":
    main()