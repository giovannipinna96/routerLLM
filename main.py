"""
Main entry point for RouterLLM system
"""

import argparse
import json
import time
import os
from typing import Dict, Any

from src.routerllm.core.system import RouterLLMSystem
from src.routerllm.data.dataset_generator import RouterDatasetGenerator
from src.routerllm.training.trainer import RouterTrainer, RouterDataset
from src.routerllm.models.router import BERTRouter
from src.routerllm.utils.logger import setup_logger


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="RouterLLM: Intelligent LLM Router System")

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Generate dataset command
    gen_parser = subparsers.add_parser('generate-data', help='Generate synthetic training dataset')
    gen_parser.add_argument('--samples', type=int, default=1200, help='Total number of samples')
    gen_parser.add_argument('--output-dir', type=str, default='./data', help='Output directory')

    # Train router command
    train_parser = subparsers.add_parser('train', help='Train the BERT router')
    train_parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    train_parser.add_argument('--model-dir', type=str, default='./models', help='Model output directory')
    train_parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    train_parser.add_argument('--use-inter-intra-loss', action='store_true', help='Use inter-intra loss function')

    # Test system command
    test_parser = subparsers.add_parser('test', help='Test the RouterLLM system')
    test_parser.add_argument('--router-type', type=str, choices=['dummy', 'bert'], default='dummy', help='Router type')
    test_parser.add_argument('--router-model', type=str, help='Path to trained router model (for BERT)')
    test_parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Config file')
    test_parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    test_parser.add_argument('--test-examples', action='store_true', help='Test with predefined examples')

    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run interactive demo')
    demo_parser.add_argument('--router-type', type=str, choices=['dummy', 'bert'], default='dummy', help='Router type')
    demo_parser.add_argument('--router-model', type=str, help='Path to trained router model (for BERT)')
    demo_parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Config file')

    args = parser.parse_args()

    if args.command == 'generate-data':
        generate_dataset(args)
    elif args.command == 'train':
        train_router(args)
    elif args.command == 'test':
        test_system(args)
    elif args.command == 'demo':
        run_demo(args)
    else:
        parser.print_help()


def generate_dataset(args):
    """Generate synthetic training dataset"""
    print("Generating synthetic dataset...")

    logger = setup_logger("dataset_generator", "./logs", "dataset_generation.log")

    generator = RouterDatasetGenerator(seed=42, logger=logger)

    dataset_info = generator.generate_and_save_complete_dataset(
        output_dir=args.output_dir,
        total_samples=args.samples
    )

    print(f"Dataset generated and saved to {args.output_dir}")
    print(f"Train samples: {len(dataset_info['train'][0])}")
    print(f"Validation samples: {len(dataset_info['val'][0])}")
    print(f"Test samples: {len(dataset_info['test'][0])}")


def train_router(args):
    """Train the BERT router"""
    print("Training BERT router...")

    logger = setup_logger("router_trainer", "./logs", "router_training.log")

    # Load datasets
    generator = RouterDatasetGenerator(logger=logger)

    train_texts, train_labels = generator.load_dataset(f"{args.data_dir}/train_dataset.json")
    val_texts, val_labels = generator.load_dataset(f"{args.data_dir}/val_dataset.json")

    # Initialize router
    router = BERTRouter(logger=logger)
    router._build_model()  # Build the model

    # Create datasets
    train_dataset = RouterDataset(train_texts, train_labels, router.tokenizer)
    val_dataset = RouterDataset(val_texts, val_labels, router.tokenizer)

    # Initialize trainer
    trainer = RouterTrainer(
        router=router,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        save_dir=args.model_dir,
        logger=logger
    )

    # Train
    history = trainer.train()

    print("Training completed!")
    print(f"Final train accuracy: {history['train_accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")

    # Save training history
    with open(f"{args.model_dir}/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)


def test_system(args):
    """Test the RouterLLM system"""
    print(f"Testing RouterLLM system with {args.router_type} router...")

    # Initialize system
    system = RouterLLMSystem(
        config_path=args.config,
        router_type=args.router_type,
        router_model_path=args.router_model,
        enable_carbon_tracking=True
    )

    system.initialize()

    if args.test_examples:
        test_with_examples(system)
    elif args.interactive:
        interactive_test(system)
    else:
        # Quick test with a single example
        quick_test(system)

    # Print system stats
    stats = system.get_system_stats()
    print("\n" + "="*60)
    print("SYSTEM STATISTICS")
    print("="*60)
    print(json.dumps(stats, indent=2))

    # Cleanup
    system.cleanup()


def test_with_examples(system: RouterLLMSystem):
    """Test system with predefined examples"""
    test_examples = [
        # Code generation examples
        "Write a Python function to calculate the factorial of a number",
        "Implement a binary search algorithm in Java",
        "Create a SQL query to find the top 10 customers by sales",

        # Text generation examples
        "Write a short story about a robot discovering emotions",
        "Create a blog post about the benefits of renewable energy",
        "Write a product description for a smart home device",

        # General purpose examples
        "Explain the difference between machine learning and artificial intelligence",
        "What are the main advantages of cloud computing?",
        "Compare the pros and cons of remote work vs office work",

        # Lightweight tasks
        "What is artificial intelligence?",
        "List 5 benefits of exercise",
        "Translate 'Hello, how are you?' to Spanish"
    ]

    print(f"\nTesting with {len(test_examples)} predefined examples...")

    for i, example in enumerate(test_examples, 1):
        print(f"\n{'='*60}")
        print(f"Example {i}/{len(test_examples)}")
        print(f"Input: {example}")
        print(f"{'='*60}")

        result = system.predict_and_generate(example)

        if result["status"] == "success":
            print(f"Predicted Model: {result['predicted_model']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Response: {result['response'][:200]}...")
            print(f"Total Time: {result['timing']['total_time']:.4f}s")
        else:
            print(f"Error: {result['error']}")

        time.sleep(1)  # Brief pause between requests


def interactive_test(system: RouterLLMSystem):
    """Interactive testing mode"""
    print("\nInteractive mode - Enter your prompts (type 'quit' to exit)")

    while True:
        try:
            user_input = input("\nEnter your prompt: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            if not user_input:
                continue

            print("Processing...")
            result = system.predict_and_generate(user_input)

            if result["status"] == "success":
                print(f"\nPredicted Model: {result['predicted_model']}")
                print(f"Confidence: {result['confidence']:.4f}")
                print(f"Response: {result['response']}")
                print(f"Total Time: {result['timing']['total_time']:.4f}s")
            else:
                print(f"Error: {result['error']}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nExiting interactive mode...")


def quick_test(system: RouterLLMSystem):
    """Quick test with a single example"""
    test_prompt = "Write a Python function to reverse a string"

    print(f"\nQuick test with prompt: '{test_prompt}'")

    result = system.predict_and_generate(test_prompt)

    if result["status"] == "success":
        print(f"Predicted Model: {result['predicted_model']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Response: {result['response']}")
        print(f"Total Time: {result['timing']['total_time']:.4f}s")
    else:
        print(f"Error: {result['error']}")


def run_demo(args):
    """Run interactive demo"""
    print("="*60)
    print("ROUTERLLM INTERACTIVE DEMO")
    print("="*60)
    print(f"Router Type: {args.router_type}")
    print(f"Config: {args.config}")
    if args.router_model:
        print(f"Router Model: {args.router_model}")
    print("="*60)

    # Initialize system
    system = RouterLLMSystem(
        config_path=args.config,
        router_type=args.router_type,
        router_model_path=args.router_model,
        enable_carbon_tracking=True
    )

    try:
        system.initialize()
        print("System initialized successfully!")

        # Show available models
        models = system.llm_manager.get_available_models()
        print(f"\nAvailable LLMs: {', '.join(models)}")

        # Run interactive test
        interactive_test(system)

    except Exception as e:
        print(f"Error initializing system: {e}")
    finally:
        # Print final stats and cleanup
        if system.is_initialized:
            stats = system.get_system_stats()
            print("\n" + "="*60)
            print("FINAL STATISTICS")
            print("="*60)
            print(f"Total Requests: {stats['total_requests']}")
            print(f"Average Inference Time: {stats['average_inference_time']:.4f}s")
            print(f"Average Loading Time: {stats['average_loading_time']:.4f}s")

            if 'carbon_footprint' in stats:
                print(f"Total Carbon Emissions: {stats['carbon_footprint']['total_emissions_kg']:.6f} kg CO2")

            print("="*60)

            system.cleanup()


if __name__ == "__main__":
    main()