#!/usr/bin/env python3
"""
Test script for complexity-based routing system

This script tests the NVIDIA and Graham complexity routers with various
types of prompts to demonstrate how they route based on complexity levels.
"""

import sys
import time
import logging
from typing import List, Dict
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from routerllm.core.system import RouterLLMSystem
from routerllm.models.router import NvidiaComplexityRouter, GrahamComplexityRouter


def setup_logging():
    """Setup logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def get_test_prompts() -> List[Dict[str, str]]:
    """
    Get a diverse set of test prompts with different complexity levels

    Returns:
        List of prompts with expected complexity levels
    """
    return [
        {
            "text": "What is 2 + 2?",
            "expected_complexity": "simple",
            "category": "Simple Math"
        },
        {
            "text": "Hello, how are you?",
            "expected_complexity": "simple",
            "category": "Greeting"
        },
        {
            "text": "Explain the concept of machine learning in simple terms.",
            "expected_complexity": "medium",
            "category": "Educational"
        },
        {
            "text": "Write a Python function to calculate factorial recursively.",
            "expected_complexity": "medium",
            "category": "Programming"
        },
        {
            "text": "Analyze the philosophical implications of quantum mechanics on free will and determinism, considering both Copenhagen and Many-worlds interpretations.",
            "expected_complexity": "complex",
            "category": "Complex Philosophy"
        },
        {
            "text": "Design a distributed system architecture for a real-time analytics platform that can handle 1 million events per second with sub-second latency, ensuring ACID compliance and fault tolerance.",
            "expected_complexity": "complex",
            "category": "Complex Engineering"
        },
        {
            "text": "Derive the mathematical relationship between entropy and information theory in the context of Shannon's work on communication systems.",
            "expected_complexity": "complex",
            "category": "Complex Math/Science"
        },
        {
            "text": "What's the weather like today?",
            "expected_complexity": "simple",
            "category": "Simple Question"
        },
        {
            "text": "Create a comprehensive business plan for a startup in the renewable energy sector, including market analysis, financial projections, and risk assessment.",
            "expected_complexity": "complex",
            "category": "Complex Business"
        },
        {
            "text": "Fix this code: print('hello world')",
            "expected_complexity": "simple",
            "category": "Simple Code Fix"
        }
    ]


def test_router_standalone(router, router_name: str, test_prompts: List[Dict], logger):
    """Test a router in standalone mode"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {router_name} Router (Standalone)")
    logger.info(f"{'='*60}")

    results = []

    for i, prompt_data in enumerate(test_prompts, 1):
        text = prompt_data["text"]
        expected = prompt_data["expected_complexity"]
        category = prompt_data["category"]

        logger.info(f"\n{i}. Testing: {category}")
        logger.info(f"Prompt: {text[:100]}{'...' if len(text) > 100 else ''}")
        logger.info(f"Expected complexity: {expected}")

        try:
            start_time = time.time()
            predicted_class, confidence = router.predict(text)
            prediction_time = time.time() - start_time

            model_name = router.get_model_name_from_class(predicted_class)

            logger.info(f"Predicted class: {predicted_class}")
            logger.info(f"Selected model: {model_name}")
            logger.info(f"Confidence/Score: {confidence:.4f}")
            logger.info(f"Prediction time: {prediction_time:.4f}s")

            results.append({
                "prompt": text,
                "category": category,
                "expected": expected,
                "predicted_class": predicted_class,
                "model_name": model_name,
                "confidence": confidence,
                "prediction_time": prediction_time
            })

        except Exception as e:
            logger.error(f"Error testing prompt {i}: {e}")
            results.append({
                "prompt": text,
                "category": category,
                "expected": expected,
                "error": str(e)
            })

    return results


def test_system_integration(router_type: str, test_prompts: List[Dict], logger):
    """Test router integrated with the full RouterLLM system"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {router_type} Router (System Integration)")
    logger.info(f"{'='*60}")

    try:
        # Initialize the system
        system = RouterLLMSystem(
            router_type=router_type,
            enable_carbon_tracking=False,  # Disable for testing
            logger=logger
        )

        system.initialize()
        logger.info("System initialized successfully")

        results = []

        for i, prompt_data in enumerate(test_prompts[:3], 1):  # Test only first 3 for integration
            text = prompt_data["text"]
            category = prompt_data["category"]

            logger.info(f"\n{i}. System test: {category}")
            logger.info(f"Prompt: {text[:100]}{'...' if len(text) > 100 else ''}")

            try:
                # Test with system integration
                result = system.predict_and_generate(
                    input_text=text,
                    max_length=100,  # Short responses for testing
                    temperature=0.7,
                    do_sample=True
                )

                if result["status"] == "success":
                    logger.info(f"Predicted model: {result['predicted_model']}")
                    logger.info(f"Confidence: {result['confidence']:.4f}")
                    logger.info(f"Total time: {result['timing']['total_time']:.4f}s")
                    logger.info(f"Response: {result['response'][:150]}{'...' if len(result['response']) > 150 else ''}")

                    results.append(result)
                else:
                    logger.error(f"Request failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                logger.error(f"Error in system integration test {i}: {e}")

        # Cleanup
        system.cleanup()
        return results

    except Exception as e:
        logger.error(f"Failed to initialize system with {router_type}: {e}")
        return []


def print_summary(nvidia_results: List, graham_results: List, logger):
    """Print a summary of the test results"""
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")

    def analyze_results(results, router_name):
        if not results:
            logger.info(f"{router_name}: No results to analyze")
            return

        logger.info(f"\n{router_name} Router Results:")
        logger.info("-" * 40)

        successful_tests = [r for r in results if "error" not in r]
        failed_tests = [r for r in results if "error" in r]

        logger.info(f"Successful tests: {len(successful_tests)}/{len(results)}")
        logger.info(f"Failed tests: {len(failed_tests)}")

        if successful_tests:
            avg_time = sum(r["prediction_time"] for r in successful_tests) / len(successful_tests)
            logger.info(f"Average prediction time: {avg_time:.4f}s")

            # Model distribution
            model_counts = {}
            for result in successful_tests:
                model = result["model_name"]
                model_counts[model] = model_counts.get(model, 0) + 1

            logger.info("Model selection distribution:")
            for model, count in model_counts.items():
                logger.info(f"  {model}: {count} times")

        if failed_tests:
            logger.info("Failed tests:")
            for result in failed_tests:
                logger.info(f"  - {result['category']}: {result['error']}")

    analyze_results(nvidia_results, "NVIDIA")
    analyze_results(graham_results, "Graham")


def main():
    """Main test function"""
    logger = setup_logging()

    logger.info("Starting Complexity-Based Routing System Tests")
    logger.info("=" * 60)

    # Get test prompts
    test_prompts = get_test_prompts()

    logger.info(f"Testing with {len(test_prompts)} diverse prompts")

    # Initialize routers for standalone testing
    nvidia_results = []
    graham_results = []

    try:
        # Test NVIDIA router standalone
        logger.info("\nInitializing NVIDIA Complexity Router...")
        nvidia_router = NvidiaComplexityRouter(logger=logger)
        nvidia_results = test_router_standalone(nvidia_router, "NVIDIA", test_prompts, logger)

    except Exception as e:
        logger.error(f"Failed to test NVIDIA router: {e}")

    try:
        # Test Graham router standalone
        logger.info("\nInitializing Graham Complexity Router...")
        graham_router = GrahamComplexityRouter(logger=logger)
        graham_results = test_router_standalone(graham_router, "Graham", test_prompts, logger)

    except Exception as e:
        logger.error(f"Failed to test Graham router: {e}")

    # Test system integration
    try:
        logger.info("\nTesting System Integration...")
        nvidia_system_results = test_system_integration("nvidia_complexity", test_prompts, logger)
        graham_system_results = test_system_integration("graham_complexity", test_prompts, logger)

    except Exception as e:
        logger.error(f"System integration tests failed: {e}")

    # Print summary
    print_summary(nvidia_results, graham_results, logger)

    logger.info("\n" + "=" * 60)
    logger.info("Complexity-Based Routing System Tests Completed")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()