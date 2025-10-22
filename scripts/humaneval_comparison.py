#!/usr/bin/env python3
"""
HumanEval Plus Comparison Script
Compares RouterLLM system (complexity-based routing) vs single large LLM
"""

import json
import time
import random
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import subprocess
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent))

from datasets import load_dataset
from src.routerllm.core.system import RouterLLMSystem
from src.routerllm.core.direct_system import DirectLLMSystem
from src.routerllm.utils.logger import setup_logger

# Code execution for validation
import tempfile
import traceback


class HumanEvalComparator:
    """Main comparator class for HumanEval Plus experiment"""

    def __init__(self,
                 config_path: str = "configs/default_config.yaml",
                 num_examples: int = 3,
                 seed: int = 42,
                 results_dir: str = "results"):
        """
        Initialize the comparator

        Args:
            config_path: Path to RouterLLM config
            num_examples: Number of HumanEval examples to test
            seed: Random seed for reproducibility
            results_dir: Directory to save results
        """
        self.config_path = config_path
        self.num_examples = num_examples
        self.seed = seed
        self.results_dir = Path(results_dir)

        # Create results directory
        self.results_dir.mkdir(exist_ok=True)

        # Setup logging
        self.logger = setup_logger(
            "humaneval_comparator",
            str(self.results_dir),
            "comparison.log"
        )

        # Set seeds
        random.seed(seed)

        # Initialize systems
        self.router_system = None
        self.direct_system = None

        # Results storage
        self.router_results = []
        self.direct_results = []
        self.selected_examples = []

    def load_humaneval_plus_dataset(self) -> List[Dict]:
        """Load HumanEval Plus dataset and select random examples"""
        self.logger.info("Loading HumanEval Plus dataset...")

        try:
            # Load the dataset
            dataset = load_dataset("evalplus/humanevalplus", split="test")
            self.logger.info(f"Loaded {len(dataset)} examples from HumanEval Plus")

            # Convert to list and shuffle
            examples = list(dataset)
            random.shuffle(examples)

            # Select random examples
            selected = examples[:self.num_examples]
            self.selected_examples = selected

            self.logger.info(f"Selected {len(selected)} random examples for testing")

            # Log selected problems
            for i, example in enumerate(selected):
                self.logger.info(f"Example {i+1}: {example['task_id']} - {example['prompt'][:100]}...")

            return selected

        except Exception as e:
            self.logger.error(f"Failed to load HumanEval Plus dataset: {e}")
            raise

    def create_coding_prompt(self, problem: Dict) -> str:
        """
        Create standardized prompt for coding problems optimized for Instruct models

        Args:
            problem: HumanEval problem dictionary

        Returns:
            Formatted prompt string
        """
        # Extract function signature and docstring from the problem
        problem_text = problem['prompt'].strip()

        # Create optimized prompt for CodeLlama-Instruct
        prompt = f"""[INST] You are a coding assistant. Complete the following Python function:

{problem_text}

Please provide only the complete function implementation. Do not include any explanations, comments, or test cases - just the working Python code. [/INST]

```python"""

        return prompt

    def _clean_generated_code(self, code: str) -> str:
        """
        Clean up generated code by removing markdown, artifacts, and formatting issues

        Args:
            code: Raw generated code

        Returns:
            Cleaned Python code
        """
        # Remove markdown code blocks
        if '```python' in code:
            # Extract code between ```python and ```
            parts = code.split('```python')
            if len(parts) > 1:
                code_part = parts[1].split('```')[0]
                code = code_part.strip()
        elif '```' in code:
            # Extract code between ``` and ```
            parts = code.split('```')
            if len(parts) >= 3:
                code = parts[1].strip()

        # Remove common prefixes/suffixes that models might add
        prefixes_to_remove = [
            "Here's the solution:",
            "Here is the solution:",
            "Solution:",
            "The answer is:",
            "```python",
            "```"
        ]

        for prefix in prefixes_to_remove:
            if code.startswith(prefix):
                code = code[len(prefix):].strip()

        # Remove trailing artifacts
        suffixes_to_remove = ["```", "</code>", "</pre>"]
        for suffix in suffixes_to_remove:
            if code.endswith(suffix):
                code = code[:-len(suffix)].strip()

        return code

    def validate_code_solution(self, code: str, problem: Dict) -> Tuple[bool, str]:
        """
        Validate if the generated code passes test cases

        Args:
            code: Generated Python code
            problem: HumanEval problem with test cases

        Returns:
            Tuple of (is_correct, error_message)
        """
        # Check if code is empty or only whitespace
        if not code or not code.strip():
            return False, "Empty or blank response"

        # Clean up the generated code (remove markdown, artifacts, etc.)
        code = self._clean_generated_code(code)

        try:
            # Create temporary file with the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Write the function definition
                f.write(code)
                f.write("\n\n")

                # Write the test cases
                test_code = problem.get('test', '')
                if test_code:
                    f.write(test_code)
                    # Add the crucial missing call to execute the tests
                    f.write(f"\n\n# Execute the validation\n")
                    f.write(f"check({problem.get('entry_point', 'candidate')})\n")
                else:
                    # Fallback: create basic test
                    f.write(f"# Test for {problem['task_id']}\n")
                    f.write("# No test cases provided\n")

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

            # Check if execution was successful
            if result.returncode == 0:
                return True, "All tests passed"
            else:
                error_msg = result.stderr or result.stdout or "Unknown error"
                return False, f"Test failed: {error_msg[:200]}"

        except subprocess.TimeoutExpired:
            return False, "Execution timeout"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def test_router_system(self) -> List[Dict]:
        """Test RouterLLM system with complexity-based routing"""
        self.logger.info("Testing RouterLLM system with complexity routing...")

        # Try complexity-based router first, fallback to dummy
        router_type = "graham_complexity"  # Using Graham complexity router
        try:
            # Initialize RouterLLM system with Graham complexity router
            self.router_system = RouterLLMSystem(
                config_path=self.config_path,
                router_type=router_type,
                enable_carbon_tracking=True
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize {router_type} router: {e}")
            self.logger.info("Falling back to dummy router...")
            router_type = "dummy"
            self.router_system = RouterLLMSystem(
                config_path=self.config_path,
                router_type=router_type,
                enable_carbon_tracking=True
            )

        self.router_system.initialize()
        results = []

        try:
            for i, problem in enumerate(self.selected_examples):
                self.logger.info(f"Router System - Processing example {i+1}/{len(self.selected_examples)}")

                # Create prompt
                prompt = self.create_coding_prompt(problem)

                # Generate response
                start_time = time.time()
                result = self.router_system.predict_and_generate(
                    input_text=prompt,
                    max_length=1024,
                    temperature=0.1,  # Low temperature for code generation
                    top_p=0.9,
                    do_sample=True
                )

                # Validate code
                if result["status"] == "success":
                    is_correct, validation_msg = self.validate_code_solution(
                        result["response"], problem
                    )
                else:
                    is_correct = False
                    validation_msg = result.get("error", "Generation failed")

                # Store result
                test_result = {
                    "example_id": i,
                    "task_id": problem["task_id"],
                    "prompt": prompt,
                    "predicted_model": result.get("predicted_model", "unknown"),
                    "predicted_class": result.get("predicted_class", -1),
                    "confidence": result.get("confidence", 0.0),
                    "response": result.get("response", ""),
                    "is_correct": is_correct,
                    "validation_message": validation_msg,
                    "timing": result.get("timing", {}),
                    "total_time": time.time() - start_time,
                    "status": result["status"]
                }

                results.append(test_result)
                self.logger.info(f"Example {i+1}: Model={test_result['predicted_model']}, "
                               f"Correct={is_correct}, Time={test_result['total_time']:.2f}s")

        finally:
            # Get system stats including carbon footprint
            stats = self.router_system.get_system_stats()

            # Store results
            self.router_results = {
                "method": "RouterLLM_ComplexityRouting",
                "router_type": router_type,  # Use actual router type
                "num_examples": len(results),
                "results": results,
                "system_stats": stats,
                "timestamp": datetime.now().isoformat()
            }

            # Cleanup
            self.router_system.cleanup()

        return results

    def test_direct_large_model(self) -> List[Dict]:
        """Test direct large LLM (StarCoder2-15B) with same prompts using dedicated DirectLLMSystem"""
        self.logger.info("Testing direct large LLM (StarCoder2-15B)...")

        # Initialize DirectLLM system with StarCoder2-15B
        large_model = "starcoder2_15b"  # Use StarCoder2-15B model

        self.direct_system = DirectLLMSystem(
            config_path=self.config_path,
            model_name=large_model,
            enable_carbon_tracking=True,
            logger=self.logger
        )

        results = []

        try:
            for i, problem in enumerate(self.selected_examples):
                self.logger.info(f"Direct LLM - Processing example {i+1}/{len(self.selected_examples)}")

                # Use same prompt as router system
                prompt = self.create_coding_prompt(problem)

                # Generate response using DirectLLMSystem
                start_time = time.time()
                response, _ = self.direct_system.process_request(
                    prompt=prompt,
                    max_new_tokens=1024,
                    temperature=0.3,  # Same temperature as router system
                    top_p=0.9,
                    do_sample=True
                )

                # Validate code
                if response:
                    is_correct, validation_msg = self.validate_code_solution(
                        response, problem
                    )
                    status = "success"
                else:
                    is_correct = False
                    validation_msg = "Generation failed"
                    status = "error"
                    response = ""

                # Store result
                test_result = {
                    "example_id": i,
                    "task_id": problem["task_id"],
                    "prompt": prompt,
                    "model_used": large_model,
                    "response": response,
                    "is_correct": is_correct,
                    "validation_message": validation_msg,
                    "total_time": time.time() - start_time,
                    "status": status
                }

                results.append(test_result)
                self.logger.info(f"Example {i+1}: Model={large_model}, "
                               f"Correct={is_correct}, Time={test_result['total_time']:.2f}s")

        finally:
            # Get system stats including carbon footprint
            stats = self.direct_system.get_system_stats()

            # Store results
            self.direct_results = {
                "method": "Direct_Large_LLM",
                "model_used": large_model,
                "num_examples": len(results),
                "results": results,
                "system_stats": stats,
                "timestamp": datetime.now().isoformat()
            }

            # Cleanup
            self.direct_system.cleanup()

        return results

    def compare_results(self) -> Dict[str, Any]:
        """Compare results between RouterLLM and direct LLM"""
        self.logger.info("Comparing results...")

        # Calculate accuracy
        router_correct = sum(1 for r in self.router_results["results"] if r["is_correct"])
        direct_correct = sum(1 for r in self.direct_results["results"] if r["is_correct"])

        router_accuracy = router_correct / len(self.router_results["results"])
        direct_accuracy = direct_correct / len(self.direct_results["results"])

        # Calculate average times
        router_avg_time = sum(r["total_time"] for r in self.router_results["results"]) / len(self.router_results["results"])
        direct_avg_time = sum(r["total_time"] for r in self.direct_results["results"]) / len(self.direct_results["results"])

        # Carbon footprint comparison
        router_carbon = self.router_results["system_stats"].get("carbon_footprint", {})
        direct_carbon = self.direct_results["system_stats"].get("carbon_footprint", {})

        router_total_co2 = router_carbon.get("total_emissions_kg", 0)
        direct_total_co2 = direct_carbon.get("total_emissions_kg", 0)

        # CO2 per correct solution
        router_co2_per_correct = router_total_co2 / max(router_correct, 1)
        direct_co2_per_correct = direct_total_co2 / max(direct_correct, 1)

        # Model usage analysis (for router system)
        model_usage = {}
        for result in self.router_results["results"]:
            model = result.get("predicted_model", "unknown")
            if model not in model_usage:
                model_usage[model] = {"count": 0, "correct": 0}
            model_usage[model]["count"] += 1
            if result["is_correct"]:
                model_usage[model]["correct"] += 1

        # Create comparison report
        comparison = {
            "experiment_info": {
                "num_examples": self.num_examples,
                "seed": self.seed,
                "timestamp": datetime.now().isoformat()
            },
            "accuracy_comparison": {
                "router_system": {
                    "correct_solutions": router_correct,
                    "total_examples": len(self.router_results["results"]),
                    "accuracy": router_accuracy
                },
                "direct_large_llm": {
                    "correct_solutions": direct_correct,
                    "total_examples": len(self.direct_results["results"]),
                    "accuracy": direct_accuracy
                },
                "accuracy_difference": router_accuracy - direct_accuracy
            },
            "performance_comparison": {
                "router_avg_time_seconds": router_avg_time,
                "direct_avg_time_seconds": direct_avg_time,
                "time_difference": router_avg_time - direct_avg_time
            },
            "carbon_footprint_comparison": {
                "router_system": {
                    "total_co2_kg": router_total_co2,
                    "co2_per_correct_solution": router_co2_per_correct,
                    "emissions_breakdown": router_carbon.get("emissions_breakdown", {})
                },
                "direct_large_llm": {
                    "total_co2_kg": direct_total_co2,
                    "co2_per_correct_solution": direct_co2_per_correct
                },
                "co2_efficiency_improvement": (direct_co2_per_correct - router_co2_per_correct) / direct_co2_per_correct if direct_co2_per_correct > 0 else 0
            },
            "router_analysis": {
                "model_usage_distribution": model_usage,
                "routing_effectiveness": "Analysis of how well complexity routing worked"
            },
            "conclusion": {
                "more_accurate": "RouterLLM" if router_accuracy > direct_accuracy else "Direct LLM" if direct_accuracy > router_accuracy else "Tie",
                "more_green": "RouterLLM" if router_co2_per_correct < direct_co2_per_correct else "Direct LLM",
                "efficiency_trade_off": "Analysis of accuracy vs environmental impact"
            }
        }

        return comparison

    def save_results(self):
        """Save all results to JSON files"""
        self.logger.info("Saving results...")

        # Save individual results
        with open(self.results_dir / "router_results.json", "w") as f:
            json.dump(self.router_results, f, indent=2)

        with open(self.results_dir / "direct_results.json", "w") as f:
            json.dump(self.direct_results, f, indent=2)

        # Save comparison analysis
        comparison = self.compare_results()
        with open(self.results_dir / "comparison_analysis.json", "w") as f:
            json.dump(comparison, f, indent=2)

        # Save selected examples for reference
        with open(self.results_dir / "selected_examples.json", "w") as f:
            json.dump(self.selected_examples, f, indent=2)

        self.logger.info(f"Results saved to {self.results_dir}")

        # Print summary
        print("\n" + "="*60)
        print("HUMANEVAL PLUS COMPARISON RESULTS")
        print("="*60)
        print(f"RouterLLM Accuracy: {comparison['accuracy_comparison']['router_system']['accuracy']:.1%}")
        print(f"Direct LLM Accuracy: {comparison['accuracy_comparison']['direct_large_llm']['accuracy']:.1%}")
        print(f"More Accurate: {comparison['conclusion']['more_accurate']}")
        print(f"More Green/Ecological: {comparison['conclusion']['more_green']}")
        print(f"RouterLLM CO2 per correct: {comparison['carbon_footprint_comparison']['router_system']['co2_per_correct_solution']:.6f} kg")
        print(f"Direct LLM CO2 per correct: {comparison['carbon_footprint_comparison']['direct_large_llm']['co2_per_correct_solution']:.6f} kg")
        print("="*60)

    def run_comparison(self):
        """Run the complete comparison experiment"""
        self.logger.info("Starting HumanEval Plus comparison experiment...")

        try:
            # Step 1: Load dataset
            self.load_humaneval_plus_dataset()

            # Step 2: Test RouterLLM system
            self.test_router_system()

            # Step 3: Test direct large LLM
            self.test_direct_large_model()

            # Step 4: Save and compare results
            self.save_results()

            self.logger.info("Comparison experiment completed successfully!")

        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            self.logger.error(traceback.format_exc())
            raise


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="HumanEval Plus Comparison")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Config file path")
    parser.add_argument("--num-examples", type=int, default=10, help="Number of examples to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--results-dir", default="results", help="Results directory")

    args = parser.parse_args()

    # Create and run comparator
    comparator = HumanEvalComparator(
        config_path=args.config,
        num_examples=args.num_examples,
        seed=args.seed,
        results_dir=args.results_dir
    )

    comparator.run_comparison()


if __name__ == "__main__":
    main()