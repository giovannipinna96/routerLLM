#!/usr/bin/env python3
"""
Test to reproduce the validation issue with by_length code
"""

import sys
import os
sys.path.append('src')
import tempfile
import subprocess

def test_validation_logic():
    """Test the problematic by_length code with proper test cases"""

    # The actual problematic code that was marked as correct
    problematic_code = '''def by_length(arr):
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
        return result'''

    # Create test cases based on the problem description
    test_cases = '''
# Test cases for by_length function
def check(candidate):
    # Test case 1: Example from problem description
    assert candidate([2, 1, 1, 4, 5, 8, 2, 3]) == ["Eight", "Five", "Four", "Three", "Two", "Two", "One", "One"], f"Test 1 failed"

    # Test case 2: Empty array
    assert candidate([]) == [], f"Test 2 failed"

    # Test case 3: Array with invalid numbers
    assert candidate([1, -1, 55]) == ["One"], f"Test 3 failed: got {candidate([1, -1, 55])}"

    print("All tests passed!")

check(by_length)
'''

    print("Testing problematic code with proper test cases...")
    print("="*60)
    print("PROBLEMATIC CODE:")
    print(problematic_code)
    print("\nTEST CASES:")
    print(test_cases)
    print("="*60)

    # Write to temporary file and execute
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(problematic_code)
            f.write("\n\n")
            f.write(test_cases)
            temp_file = f.name

        # Execute the code
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Clean up
        os.unlink(temp_file)

        print("SUBPROCESS RESULT:")
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")

        if result.returncode == 0:
            print("❌ VALIDATION BUG: Code marked as PASSED but it's clearly wrong!")
        else:
            print("✅ Validation correctly identified the error")

        return result.returncode == 0

    except Exception as e:
        print(f"Test execution failed: {e}")
        return False

def test_correct_implementation():
    """Test with correct implementation to verify test cases work"""

    correct_code = '''def by_length(arr):
    # Filter numbers between 1-9, sort, reverse, then convert to names
    valid_nums = [x for x in arr if 1 <= x <= 9]
    sorted_nums = sorted(valid_nums)
    reversed_nums = sorted_nums[::-1]

    names = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
             6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}

    return [names[num] for num in reversed_nums]'''

    test_cases = '''
# Test cases for by_length function
def check(candidate):
    # Test case 1: Example from problem description
    assert candidate([2, 1, 1, 4, 5, 8, 2, 3]) == ["Eight", "Five", "Four", "Three", "Two", "Two", "One", "One"], f"Test 1 failed"

    # Test case 2: Empty array
    assert candidate([]) == [], f"Test 2 failed"

    # Test case 3: Array with invalid numbers
    assert candidate([1, -1, 55]) == ["One"], f"Test 3 failed: got {candidate([1, -1, 55])}"

    print("All tests passed!")

check(by_length)
'''

    print("\n" + "="*60)
    print("TESTING CORRECT IMPLEMENTATION:")
    print("="*60)

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(correct_code)
            f.write("\n\n")
            f.write(test_cases)
            temp_file = f.name

        # Execute the code
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Clean up
        os.unlink(temp_file)

        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")

        if result.returncode == 0:
            print("✅ Correct implementation passes all tests!")
        else:
            print("❌ Test cases might be wrong")

    except Exception as e:
        print(f"Test execution failed: {e}")

def main():
    print("VALIDATION SYSTEM BUG REPRODUCTION TEST")
    print("="*60)

    # Test the problematic code that was marked as correct
    problematic_passed = test_validation_logic()

    # Test with correct implementation
    test_correct_implementation()

    print("\n" + "="*60)
    print("CONCLUSION:")
    if problematic_passed:
        print("❌ VALIDATION BUG CONFIRMED: Wrong code passes validation")
        print("The validation system has a serious flaw!")
    else:
        print("✅ Validation works correctly - problem might be elsewhere")

if __name__ == "__main__":
    main()