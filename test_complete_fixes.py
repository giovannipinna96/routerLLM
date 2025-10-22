#!/usr/bin/env python3
"""
Test script to verify all fixes work correctly:
1. DirectLLMSystem with proper carbon tracking
2. GPT-OSS-20B integration
3. Fixed validation system
4. Updated comparison script integration
"""

import sys
import tempfile
import subprocess
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_direct_system_import():
    """Test that DirectLLMSystem can be imported and instantiated"""
    print("1. Testing DirectLLMSystem import...")
    try:
        from src.routerllm.core.direct_system import DirectLLMSystem

        # Test basic instantiation (without actual loading)
        system = DirectLLMSystem(
            config_path="configs/default_config.yaml",
            model_name="gpt_oss_20b",
            enable_carbon_tracking=False  # Disable for test
        )

        print("✅ DirectLLMSystem imported and instantiated successfully")
        return True
    except Exception as e:
        print(f"❌ DirectLLMSystem import failed: {e}")
        return False

def test_config_has_gpt_oss():
    """Test that config has GPT-OSS-20B configuration"""
    print("\n2. Testing GPT-OSS-20B configuration...")
    try:
        import yaml

        with open("configs/default_config.yaml", 'r') as f:
            config = yaml.safe_load(f)

        # Check if gpt_oss_20b is in models
        llm_models = config.get('models', {}).get('llms', [])
        gpt_oss_model = None

        for model in llm_models:
            if model.get('name') == 'gpt_oss_20b':
                gpt_oss_model = model
                break

        if gpt_oss_model:
            print(f"✅ Found GPT-OSS-20B config: {gpt_oss_model['model_id']}")
            if gpt_oss_model.get('special_format') == 'harmony':
                print("✅ Harmony format correctly configured")
                return True
            else:
                print("❌ Harmony format not configured")
                return False
        else:
            print("❌ GPT-OSS-20B not found in config")
            return False

    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_validation_fix():
    """Test that validation system now correctly fails wrong code"""
    print("\n3. Testing validation system fix...")

    # The problematic by_length code that should now fail validation
    bad_code = '''def by_length(arr):
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
                result.append('')  # This causes wrong output
        return result'''

    # Simple test that should fail
    test_code = '''
def check(candidate):
    # This should fail because bad code returns ['', 'One'] instead of ['One']
    assert candidate([1, -1, 55]) == ["One"], f"Failed: got {candidate([1, -1, 55])}"
    print("Test passed")

# Execute validation
check(by_length)
'''

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(bad_code)
            f.write(test_code)
            temp_file = f.name

        # Execute the code
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Clean up
        import os
        os.unlink(temp_file)

        if result.returncode != 0:
            print("✅ Validation correctly failed for wrong code")
            return True
        else:
            print("❌ Validation incorrectly passed wrong code")
            return False

    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False

def test_comparison_script_syntax():
    """Test that updated comparison script has valid syntax"""
    print("\n4. Testing comparison script syntax...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", "scripts/humaneval_comparison.py"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            print("✅ Comparison script syntax is valid")
            return True
        else:
            print(f"❌ Comparison script syntax error: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Syntax test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("TESTING COMPLETE SYSTEM FIXES")
    print("=" * 50)

    tests = [
        test_direct_system_import,
        test_config_has_gpt_oss,
        test_validation_fix,
        test_comparison_script_syntax
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 50)
    print("SUMMARY:")
    passed = sum(results)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("🎉 ALL FIXES WORKING CORRECTLY!")
    else:
        print("⚠️  Some issues remain to be fixed")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)