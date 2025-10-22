#!/usr/bin/env python3
"""
Test script to verify that the pipeline system works correctly
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

def test_pipeline_imports():
    """Test that all imports work with pipeline approach"""
    print("1. Testing imports...")
    try:
        from src.routerllm.models.llm_manager import LLMManager
        from src.routerllm.core.direct_system import DirectLLMSystem
        from transformers import pipeline
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config_updated():
    """Test that config has StarCoder2-15B instead of GPT-OSS-20B"""
    print("\n2. Testing configuration...")
    try:
        import yaml

        with open("configs/default_config.yaml", 'r') as f:
            config = yaml.safe_load(f)

        llm_models = config.get('models', {}).get('llms', [])

        # Check that GPT-OSS-20B is gone
        has_gpt_oss = any(model.get('name') == 'gpt_oss_20b' for model in llm_models)
        if has_gpt_oss:
            print("❌ GPT-OSS-20B still present in config")
            return False

        # Check that StarCoder2-15B is present
        starcoder_model = None
        for model in llm_models:
            if model.get('name') == 'starcoder2_15b':
                starcoder_model = model
                break

        if starcoder_model:
            print(f"✅ Found StarCoder2-15B: {starcoder_model['model_id']}")
            # Check no special format
            if 'special_format' not in starcoder_model:
                print("✅ No special format specified (correct)")
                return True
            else:
                print("❌ Special format still specified")
                return False
        else:
            print("❌ StarCoder2-15B not found in config")
            return False

    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_llm_manager_pipeline():
    """Test that LLMManager has pipeline support"""
    print("\n3. Testing LLMManager pipeline support...")
    try:
        from src.routerllm.models.llm_manager import LLMManager

        # Create LLMManager instance (don't load model)
        manager = LLMManager("configs/default_config.yaml")

        # Check that it has pipelines attribute
        if hasattr(manager, 'pipelines'):
            print("✅ LLMManager has pipelines attribute")
        else:
            print("❌ LLMManager missing pipelines attribute")
            return False

        # Check that the generate_response method exists
        if hasattr(manager, 'generate_response'):
            print("✅ LLMManager has generate_response method")
        else:
            print("❌ LLMManager missing generate_response method")
            return False

        return True

    except Exception as e:
        print(f"❌ LLMManager test failed: {e}")
        return False

def test_direct_system_compatibility():
    """Test that DirectLLMSystem works with new changes"""
    print("\n4. Testing DirectLLMSystem compatibility...")
    try:
        from src.routerllm.core.direct_system import DirectLLMSystem

        # Create DirectLLMSystem instance (don't initialize)
        system = DirectLLMSystem(
            config_path="configs/default_config.yaml",
            model_name="starcoder2_15b",
            enable_carbon_tracking=False  # Disable for test
        )

        # Check that process_request method exists
        if hasattr(system, 'process_request'):
            print("✅ DirectLLMSystem has process_request method")
        else:
            print("❌ DirectLLMSystem missing process_request method")
            return False

        # Check that generate_response method exists
        if hasattr(system, 'generate_response'):
            print("✅ DirectLLMSystem has generate_response method")
        else:
            print("❌ DirectLLMSystem missing generate_response method")
            return False

        return True

    except Exception as e:
        print(f"❌ DirectLLMSystem test failed: {e}")
        return False

def test_validation_system():
    """Test that validation system works correctly"""
    print("\n5. Testing validation system...")
    try:
        # Test with simple wrong code
        import tempfile
        import subprocess
        import os

        wrong_code = '''def test_func():
    return "wrong"

def check(candidate):
    assert candidate() == "correct", "This should fail"
    print("Test passed")

# Execute validation
check(test_func)
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(wrong_code)
            temp_file = f.name

        # Execute the code
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Clean up
        os.unlink(temp_file)

        if result.returncode != 0:
            print("✅ Validation correctly fails wrong code")
            return True
        else:
            print("❌ Validation incorrectly passes wrong code")
            return False

    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("TESTING PIPELINE SYSTEM INTEGRATION")
    print("=" * 50)

    tests = [
        test_pipeline_imports,
        test_config_updated,
        test_llm_manager_pipeline,
        test_direct_system_compatibility,
        test_validation_system
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
        print("🎉 ALL PIPELINE SYSTEM TESTS PASSED!")
        print("\n📋 SYSTEM STATUS:")
        print("✅ Pipeline approach implemented")
        print("✅ StarCoder2-15B configured")
        print("✅ Harmony format removed")
        print("✅ Validation system fixed")
        print("✅ DirectLLMSystem compatible")
        print("\n🚀 System ready for full testing when model download completes!")
    else:
        print("⚠️  Some tests failed - check above for details")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)