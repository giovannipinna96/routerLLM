#!/usr/bin/env python3
"""
Quick test to verify Graham complexity router works
"""

import sys
import os
sys.path.append('src')

from routerllm.models.router import GrahamComplexityRouter

def test_graham_router():
    print("Testing Graham Complexity Router...")

    try:
        # Initialize router
        router = GrahamComplexityRouter()
        print("✓ Router initialized successfully")

        # Test different complexity examples
        test_cases = [
            "def hello(): return 'hello'",  # Should be easy
            "Write a function to check if a string is a palindrome",  # Should be medium
            "Implement a complex algorithm to solve the traveling salesman problem with dynamic programming and memoization",  # Should be hard
        ]

        for i, test in enumerate(test_cases):
            predicted_class, confidence = router.predict(test)
            model_name = router.get_model_name_from_class(predicted_class)
            print(f"Test {i+1}: '{test[:50]}...'")
            print(f"  → Class: {predicted_class}, Model: {model_name}, Confidence: {confidence:.4f}")
            print()

        print("✓ Graham router test completed successfully!")
        return True

    except Exception as e:
        print(f"✗ Graham router test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_graham_router()
    sys.exit(0 if success else 1)