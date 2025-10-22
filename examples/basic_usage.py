"""
Basic usage examples for RouterLLM
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.routerllm.core.system import RouterLLMSystem
from src.routerllm.data.dataset_generator import RouterDatasetGenerator


def example_dummy_router():
    """Example using dummy router"""
    print("=" * 60)
    print("EXAMPLE: Dummy Router")
    print("=" * 60)

    # Initialize system with dummy router
    system = RouterLLMSystem(
        config_path="configs/default_config.yaml",
        router_type="dummy",
        enable_carbon_tracking=True
    )

    # Initialize the system
    system.initialize()

    # Test with different types of prompts
    test_prompts = [
        "Write a Python function to calculate fibonacci numbers",
        "Explain the concept of machine learning",
        "What is the capital of France?",
        "Create a story about a time traveler"
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: {prompt}")
        print("-" * 40)

        result = system.predict_and_generate(
            prompt,
            max_length=200,
            temperature=0.7
        )

        if result["status"] == "success":
            print(f"Selected Model: {result['predicted_model']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Response: {result['response'][:150]}...")
            print(f"Time: {result['timing']['total_time']:.4f}s")
        else:
            print(f"Error: {result['error']}")

    # Print system statistics
    stats = system.get_system_stats()
    print(f"\nTotal Requests: {stats['total_requests']}")
    print(f"Average Inference Time: {stats['average_inference_time']:.4f}s")

    # Cleanup
    system.cleanup()


def example_dataset_generation():
    """Example of generating synthetic dataset"""
    print("=" * 60)
    print("EXAMPLE: Dataset Generation")
    print("=" * 60)

    # Initialize generator
    generator = RouterDatasetGenerator(seed=42)

    # Generate a small dataset
    texts, labels = generator.generate_dataset(total_samples=50, balance_classes=True)

    print(f"Generated {len(texts)} samples")

    # Show some examples
    print("\nSample data:")
    for i in range(min(5, len(texts))):
        category_name = generator.categories[labels[i]]["name"]
        print(f"Class {labels[i]} ({category_name}): {texts[i]}")

    # Save dataset
    generator.save_dataset(texts, labels, "examples/sample_dataset.json")
    print("\nDataset saved to examples/sample_dataset.json")


def example_batch_processing():
    """Example of batch processing"""
    print("=" * 60)
    print("EXAMPLE: Batch Processing")
    print("=" * 60)

    # Initialize system
    system = RouterLLMSystem(
        config_path="configs/default_config.yaml",
        router_type="dummy",
        enable_carbon_tracking=True
    )

    system.initialize()

    # Batch of prompts
    batch_prompts = [
        "Write a function to sort a list",
        "Explain quantum computing",
        "What is photosynthesis?",
        "Create a haiku about technology",
        "List the planets in our solar system"
    ]

    print(f"Processing batch of {len(batch_prompts)} prompts...")

    # Process batch
    results = system.batch_process(
        batch_prompts,
        max_length=150,
        temperature=0.8
    )

    # Display results
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        if result["status"] == "success":
            print(f"  Model: {result['predicted_model']}")
            print(f"  Response: {result['response'][:100]}...")
        else:
            print(f"  Error: {result['error']}")

    # Show final stats
    stats = system.get_system_stats()
    print(f"\nBatch processed in total time: {sum(r['timing']['total_time'] for r in results if 'timing' in r):.4f}s")

    system.cleanup()


def example_carbon_tracking():
    """Example focusing on carbon footprint tracking"""
    print("=" * 60)
    print("EXAMPLE: Carbon Footprint Tracking")
    print("=" * 60)

    # Initialize system with carbon tracking
    system = RouterLLMSystem(
        config_path="configs/default_config.yaml",
        router_type="dummy",
        enable_carbon_tracking=True
    )

    system.initialize()

    # Test prompts with different complexities
    test_prompts = [
        "What is 2+2?",  # Simple
        "Write a detailed explanation of neural networks",  # Complex
        "Create a Python script for data analysis"  # Medium
    ]

    for prompt in test_prompts:
        print(f"\nProcessing: {prompt[:50]}...")
        result = system.predict_and_generate(prompt, max_length=300)

        if result["status"] == "success":
            print(f"Model: {result['predicted_model']}")
            print(f"Time: {result['timing']['total_time']:.4f}s")

    # Show carbon footprint breakdown
    stats = system.get_system_stats()
    if 'carbon_footprint' in stats:
        print("\nCarbon Footprint Breakdown:")
        breakdown = stats['carbon_footprint']['emissions_breakdown']
        for component, emissions in breakdown.items():
            print(f"  {component}: {emissions:.6f} kg CO2")
        print(f"Total: {stats['carbon_footprint']['total_emissions_kg']:.6f} kg CO2")

    system.cleanup()


def example_router_switching():
    """Example of switching between router types"""
    print("=" * 60)
    print("EXAMPLE: Router Switching")
    print("=" * 60)

    # Initialize with dummy router
    system = RouterLLMSystem(
        config_path="configs/default_config.yaml",
        router_type="dummy",
        enable_carbon_tracking=False  # Disable for this example
    )

    system.initialize()

    test_prompt = "Write a Python function to find prime numbers"

    # Test with dummy router
    print("\n1. Using Dummy Router:")
    result1 = system.predict_and_generate(test_prompt)
    if result1["status"] == "success":
        print(f"   Model: {result1['predicted_model']}")
        print(f"   Confidence: {result1['confidence']:.4f}")

    # Switch to BERT router (Note: would need trained model)
    print("\n2. Switching to BERT Router (demo - no trained model):")
    try:
        system.switch_router("bert")
        print("   Router switched successfully (but no trained model available)")
    except Exception as e:
        print(f"   Expected error: {e}")

    # Switch back to dummy
    system.switch_router("dummy")
    print("\n3. Switched back to Dummy Router")

    system.cleanup()


if __name__ == "__main__":
    print("RouterLLM Examples")
    print("=" * 60)

    # Run examples
    try:
        example_dataset_generation()
        print("\n" + "=" * 60 + "\n")

        example_dummy_router()
        print("\n" + "=" * 60 + "\n")

        example_batch_processing()
        print("\n" + "=" * 60 + "\n")

        example_carbon_tracking()
        print("\n" + "=" * 60 + "\n")

        example_router_switching()

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)