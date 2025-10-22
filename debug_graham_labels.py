#!/usr/bin/env python3
"""
Debug Graham complexity router labels
"""

import sys
sys.path.append('src')

from transformers import pipeline

def debug_graham_labels():
    print("Debugging Graham Complexity Router labels...")

    try:
        # Initialize the pipeline directly
        classifier = pipeline(
            "text-classification",
            model="grahamaco/question-complexity-classifier",
            device=-1  # Use CPU
        )
        print("✓ Classifier initialized successfully")

        # Test different complexity examples
        test_cases = [
            "What is 2 + 2?",  # Should be simple/easy
            "Write a function to check if a string is a palindrome",  # Should be medium
            "Implement a complex algorithm to solve the traveling salesman problem with dynamic programming",  # Should be hard/complex
        ]

        for i, test in enumerate(test_cases):
            results = classifier(test)
            print(f"Test {i+1}: '{test[:50]}...'")
            print(f"  Raw results: {results}")
            if isinstance(results, list) and len(results) > 0:
                for result in results:
                    print(f"    Label: {result['label']}, Score: {result['score']:.4f}")
            print()

        print("✓ Graham label debug completed!")
        return True

    except Exception as e:
        print(f"✗ Graham label debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_graham_labels()
    sys.exit(0 if success else 1)