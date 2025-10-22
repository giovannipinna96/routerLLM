#!/usr/bin/env python3
"""
Debug script to test LLM generation in isolation
"""

import sys
import os
sys.path.append('src')

from routerllm.core.system import RouterLLMSystem
from routerllm.models.llm_manager import LLMManager
import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_generation(model_name: str):
    """Test individual model generation"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing model: {model_name}")
    logger.info(f"{'='*60}")

    try:
        # Initialize LLM Manager
        llm_manager = LLMManager('configs/default_config.yaml', logger=logger)

        # Simple test prompts
        test_prompts = [
            # Very simple prompt
            "Hello, how are you?",

            # Simple coding prompt
            "[INST] Write a Python function that adds two numbers. [/INST]\n\n```python",

            # HumanEval style prompt (simplified)
            "[INST] Complete this Python function:\n\ndef add(a, b):\n    \"\"\"Add two numbers\"\"\"\n    # Your code here\n\nProvide only the function code. [/INST]\n\n```python"
        ]

        for i, prompt in enumerate(test_prompts):
            logger.info(f"\n--- Test {i+1}: {prompt[:50]}... ---")

            try:
                # Load model
                if not llm_manager.load_model(model_name):
                    logger.error(f"Failed to load model {model_name}")
                    continue

                # Generate response
                response = llm_manager.generate_response(
                    prompt=prompt,
                    max_new_tokens=256,
                    temperature=0.1,  # Very deterministic
                    do_sample=False   # Greedy decoding
                )

                if response:
                    logger.info(f"✓ SUCCESS - Generated {len(response)} chars")
                    logger.info(f"Response: '{response[:200]}{'...' if len(response) > 200 else ''}'")
                else:
                    logger.error(f"✗ FAILED - Empty response")

            except Exception as e:
                logger.error(f"✗ ERROR - {e}")

        # Unload model
        llm_manager.unload_current_model()

    except Exception as e:
        logger.error(f"Model {model_name} test failed: {e}")
        import traceback
        traceback.print_exc()

def test_all_models():
    """Test all configured models"""
    # Model list in order of complexity (lightest first)
    models_to_test = [
        "phi3_mini",      # Lightest model first
        "codellama_7b",   # Medium model
        "mistral_7b",     # Medium model alternative
        "codellama_13b"   # Largest model last
    ]

    for model_name in models_to_test:
        test_model_generation(model_name)
        logger.info(f"\nCompleted testing {model_name}")
        logger.info("-" * 60)

def test_router_integration():
    """Test RouterLLM system with Graham router"""
    logger.info(f"\n{'='*60}")
    logger.info("Testing RouterLLM System Integration")
    logger.info(f"{'='*60}")

    try:
        # Initialize RouterLLM system
        system = RouterLLMSystem(
            config_path="configs/default_config.yaml",
            router_type="graham_complexity",
            enable_carbon_tracking=False  # Disable for faster testing
        )

        # Simple test prompt
        test_prompt = "[INST] Write a Python function that checks if a number is even. [/INST]\n\n```python"

        logger.info(f"Test prompt: {test_prompt}")

        # Generate response
        result = system.predict_and_generate(
            input_text=test_prompt,
            max_length=256,
            temperature=0.1
        )

        logger.info(f"Router selected: {result.get('predicted_model', 'unknown')}")
        logger.info(f"Confidence: {result.get('confidence', 0.0):.4f}")
        logger.info(f"Response length: {len(result.get('response', ''))}")
        logger.info(f"Response: '{result.get('response', '')[:200]}...'")

        if result.get('response'):
            logger.info("✓ RouterLLM integration SUCCESS")
        else:
            logger.error("✗ RouterLLM integration FAILED - Empty response")

    except Exception as e:
        logger.error(f"RouterLLM integration test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main debug function"""
    logger.info("Starting LLM Generation Debug Session")

    if len(sys.argv) > 1:
        # Test specific model
        model_name = sys.argv[1]
        test_model_generation(model_name)
    else:
        # Test all models
        test_all_models()

        # Test router integration
        test_router_integration()

    logger.info("\nDebug session completed!")

if __name__ == "__main__":
    main()