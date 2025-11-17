#!/usr/bin/env python3
"""
Test Router and Save Results Script

This script tests a RouterLLM router with predefined examples,
collects statistics including carbon footprint, and saves results to JSON.

Usage:
    python test_router_and_save.py --router-type bert --router-model ./models/best_router.pt --output-file results.json
    python test_router_and_save.py --router-type dummy --output-file results.json
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
import time

# Add src to path - ensure we have the correct project root
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.routerllm.core.system import RouterLLMSystem
from src.routerllm.utils.logger import setup_logger


# Predefined test examples covering different complexities
TEST_PROMPTS = [
    # Category 0: Complex code generation (Professional level)
    "Write a Python function to implement a distributed MapReduce framework with fault tolerance and data partitioning",
    "Implement a neural network from scratch in Python with backpropagation, batch normalization, and dropout",
    "Create a C++ template metaprogramming solution for compile-time matrix operations with SIMD optimization",

    # Category 2: Expert level code tasks
    "Write a Python function to calculate factorial using recursion",
    "Implement binary search algorithm in JavaScript with edge case handling",
    "Create a REST API endpoint in Python Flask for user authentication with JWT tokens",

    # Category 3: Beginner level code tasks
    "Write a simple Python function to reverse a string",
    "How do I print 'Hello World' in Python?",
    "Convert temperature from Celsius to Fahrenheit in Python",
    "Write a function to check if a number is even or odd",

    # Mixed complexity
    "Explain how to use list comprehensions in Python",
    "What is the difference between a list and a tuple in Python?",
    "Implement a quick sort algorithm with detailed comments",
    "Write a SQL query to join two tables and filter results",
    "Create a simple calculator program in Python with error handling",
]


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Test RouterLLM router and save results to JSON"
    )

    parser.add_argument(
        '--router-type',
        type=str,
        required=True,
        choices=['dummy', 'bert', 'graham_complexity', 'moe', 'rl', 'integrated'],
        help='Type of router to test'
    )

    parser.add_argument(
        '--router-model',
        type=str,
        default=None,
        help='Path to trained router model (required for BERT, MoE, RL)'
    )

    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='Output JSON file path for results'
    )

    parser.add_argument(
        '--router-name',
        type=str,
        default=None,
        help='Human-readable name for this router (for reporting)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Configuration file path'
    )

    parser.add_argument(
        '--test-examples',
        action='store_true',
        default=True,
        help='Use predefined test examples'
    )

    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='Maximum generation length'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Generation temperature'
    )

    return parser.parse_args()


def test_router(args):
    """
    Test router with predefined examples and collect statistics

    Returns:
        dict: Results dictionary with stats and carbon footprint
    """
    logger = setup_logger("test_router", level="INFO")

    # Determine router name
    router_name = args.router_name or args.router_type.upper()

    logger.info(f"Testing router: {router_name}")
    logger.info(f"Router type: {args.router_type}")
    if args.router_model:
        logger.info(f"Router model: {args.router_model}")

    # Initialize system
    try:
        system = RouterLLMSystem(
            config_path=args.config,
            router_type=args.router_type,
            router_model_path=args.router_model,
            enable_carbon_tracking=True,  # Always enable carbon tracking
            logger=logger
        )

        system.initialize()
        logger.info("System initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise

    # Test prompts
    test_prompts = TEST_PROMPTS if args.test_examples else []

    if not test_prompts:
        logger.warning("No test prompts provided")
        return None

    logger.info(f"Testing with {len(test_prompts)} prompts")

    # Collect results
    results = {
        "router_name": router_name,
        "router_type": args.router_type,
        "router_model": args.router_model,
        "config_file": args.config,
        "test_timestamp": datetime.now().isoformat(),
        "num_test_prompts": len(test_prompts),
        "test_results": [],
        "summary_stats": {},
        "carbon_footprint": {}
    }

    # Test each prompt
    total_time = 0
    total_router_time = 0
    total_generation_time = 0
    model_selections = {}

    for i, prompt in enumerate(test_prompts, 1):
        logger.info(f"Testing prompt {i}/{len(test_prompts)}: {prompt[:60]}...")

        try:
            start_time = time.time()

            result = system.predict_and_generate(
                input_text=prompt,
                max_length=args.max_length,
                temperature=args.temperature
            )

            end_time = time.time()
            elapsed_time = end_time - start_time

            if result["status"] == "success":
                # Record successful result
                test_result = {
                    "prompt": prompt,
                    "predicted_model": result["predicted_model"],
                    "confidence": result["confidence"],
                    "response": result["response"],  # Full response
                    "response_length": len(result["response"]),  # Character count
                    "timing": {
                        "total_time": result["timing"]["total_time"],
                        "router_time": result["timing"]["router_time"],
                        "loading_time": result["timing"]["loading_time"],
                        "generation_time": result["timing"]["generation_time"]
                    },
                    "status": "success"
                }

                # Aggregate stats
                total_time += result["timing"]["total_time"]
                total_router_time += result["timing"]["router_time"]
                total_generation_time += result["timing"]["generation_time"]

                # Count model selections
                model = result["predicted_model"]
                model_selections[model] = model_selections.get(model, 0) + 1

            else:
                # Record failed result
                test_result = {
                    "prompt": prompt,
                    "error": result.get("error", "Unknown error"),
                    "status": "failed"
                }

            results["test_results"].append(test_result)

        except Exception as e:
            logger.error(f"Error testing prompt {i}: {e}")
            test_result = {
                "prompt": prompt,
                "error": str(e),
                "status": "failed"
            }
            results["test_results"].append(test_result)

    # Calculate summary statistics
    num_successful = sum(1 for r in results["test_results"] if r["status"] == "success")
    num_failed = len(test_prompts) - num_successful

    results["summary_stats"] = {
        "total_prompts": len(test_prompts),
        "successful": num_successful,
        "failed": num_failed,
        "success_rate": num_successful / len(test_prompts) if test_prompts else 0,
        "average_total_time": total_time / num_successful if num_successful > 0 else 0,
        "average_router_time": total_router_time / num_successful if num_successful > 0 else 0,
        "average_generation_time": total_generation_time / num_successful if num_successful > 0 else 0,
        "model_selections": model_selections,
        "most_selected_model": max(model_selections, key=model_selections.get) if model_selections else None
    }

    # Get system stats (includes carbon footprint)
    try:
        system_stats = system.get_system_stats()

        if "carbon_footprint" in system_stats:
            results["carbon_footprint"] = {
                "total_emissions_kg": system_stats["carbon_footprint"]["total_emissions_kg"],
                "emissions_breakdown": system_stats["carbon_footprint"]["emissions_breakdown"],
                "average_emissions_per_request": (
                    system_stats["carbon_footprint"]["total_emissions_kg"] / num_successful
                    if num_successful > 0 else 0
                )
            }

        # Additional system stats
        results["system_info"] = {
            "total_requests_processed": system_stats.get("total_requests", 0),
            "average_inference_time": system_stats.get("average_inference_time", 0),
            "current_model": system_stats.get("current_model", None)
        }

    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        results["carbon_footprint"] = {"error": str(e)}

    # Cleanup
    try:
        system.cleanup()
        logger.info("System cleanup completed")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

    return results


def main():
    """Main entry point"""
    args = parse_args()

    print(f"\n{'='*60}")
    print("RouterLLM Router Testing Script")
    print(f"{'='*60}")
    print(f"Router: {args.router_name or args.router_type}")
    print(f"Output: {args.output_file}")
    print(f"{'='*60}\n")

    # Run tests
    try:
        results = test_router(args)

        if results is None:
            print("ERROR: No results generated")
            sys.exit(1)

        # Save results to JSON
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print("Testing Completed Successfully")
        print(f"{'='*60}")
        print(f"Results saved to: {args.output_file}")
        print(f"\nSummary:")
        print(f"  Total prompts: {results['summary_stats']['total_prompts']}")
        print(f"  Successful: {results['summary_stats']['successful']}")
        print(f"  Failed: {results['summary_stats']['failed']}")
        print(f"  Success rate: {results['summary_stats']['success_rate']*100:.1f}%")
        print(f"  Avg total time: {results['summary_stats']['average_total_time']:.3f}s")
        print(f"  Most selected model: {results['summary_stats']['most_selected_model']}")

        if "total_emissions_kg" in results["carbon_footprint"]:
            print(f"\nCarbon Footprint:")
            print(f"  Total emissions: {results['carbon_footprint']['total_emissions_kg']:.6f} kg CO2")
            print(f"  Avg per request: {results['carbon_footprint']['average_emissions_per_request']:.6f} kg CO2")

        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
