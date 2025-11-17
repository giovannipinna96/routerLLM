#!/usr/bin/env python3
"""
Large Model Benchmark Script
Dedicated tool for testing 150B+ parameter models on HumanEval Plus dataset
Optimized for 2-3 GPU setups with comprehensive pre-flight checks
"""

import json
import time
import random
import logging
import os
import argparse
import sys
import torch
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import tempfile

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent))

from datasets import load_dataset
from src.routerllm.models.large_model_manager import DirectLargeModelSystem, LargeModelManager
from src.routerllm.utils.logger import setup_logger


class LargeModelBenchmark:
    """
    Dedicated benchmark tool for testing large language models (150B+)
    Includes comprehensive pre-flight checks and progressive result saving
    """

    AVAILABLE_MODELS = {
        "llama3_405b": {
            "name": "Llama-3.1-405B",
            "params": "405B",
            "min_vram_gb": 200,  # With 4-bit quantization
            "recommended_gpus": "3-4 x A100 80GB",
            "loading_time_min": "15-25"
        },
        "falcon_180b": {
            "name": "Falcon-180B",
            "params": "180B",
            "min_vram_gb": 150,
            "recommended_gpus": "2-3 x A100 80GB",
            "loading_time_min": "10-20"
        },
        "bloom_176b": {
            "name": "BLOOM-176B",
            "params": "176B",
            "min_vram_gb": 150,
            "recommended_gpus": "2-3 x A100 80GB",
            "loading_time_min": "10-20"
        },
        "codellama_70b": {
            "name": "CodeLlama-70B",
            "params": "70B",
            "min_vram_gb": 80,
            "recommended_gpus": "2 x A100 40GB or 1 x A100 80GB",
            "loading_time_min": "5-10"
        }
    }

    def __init__(
        self,
        config_path: str,
        model_choice: str,
        num_examples: int,
        skip_execution: bool,
        results_dir: str,
        seed: int = 42
    ):
        """
        Initialize the large model benchmark

        Args:
            config_path: Path to configuration file
            model_choice: Which large model to use (llama3_405b, falcon_180b, etc.)
            num_examples: Number of HumanEval examples to test
            skip_execution: Skip code execution validation
            results_dir: Directory to save results
            seed: Random seed
        """
        self.config_path = config_path
        self.model_choice = model_choice
        self.num_examples = num_examples
        self.skip_execution = skip_execution
        self.results_dir = Path(results_dir)
        self.seed = seed

        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = setup_logger(
            "large_model_benchmark",
            str(self.results_dir),
            f"benchmark_{model_choice}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        # Set seed
        random.seed(seed)
        torch.manual_seed(seed)

        # Components
        self.large_model_system = None
        self.selected_examples = []
        self.results = []

        self.logger.info("="*80)
        self.logger.info("LARGE MODEL BENCHMARK INITIALIZED")
        self.logger.info("="*80)
        self.logger.info(f"Model: {self.model_choice} ({self.AVAILABLE_MODELS[model_choice]['name']})")
        self.logger.info(f"Examples: {num_examples}")
        self.logger.info(f"Skip Execution: {skip_execution}")
        self.logger.info(f"Results Dir: {results_dir}")
        self.logger.info("="*80)

    def run_preflight_checks(self) -> bool:
        """
        Run comprehensive pre-flight checks before loading model

        Returns:
            True if all checks pass, False otherwise
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("🔍 PRE-FLIGHT CHECKS")
        self.logger.info("="*80)

        all_checks_passed = True
        model_info = self.AVAILABLE_MODELS[self.model_choice]

        # Check 1: GPU Availability
        self.logger.info("\n📊 Check 1/5: GPU Availability")
        if not torch.cuda.is_available():
            self.logger.error("❌ FAILED: No CUDA GPUs detected")
            all_checks_passed = False
        else:
            num_gpus = torch.cuda.device_count()
            self.logger.info(f"✓ PASSED: {num_gpus} GPU(s) available")

            total_vram_gb = 0
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024**3)
                total_vram_gb += vram_gb
                self.logger.info(f"  GPU {i}: {props.name} - {vram_gb:.1f}GB VRAM")

            # Check total VRAM
            min_vram = model_info["min_vram_gb"]
            if total_vram_gb < min_vram:
                self.logger.warning(f"⚠️  WARNING: Total VRAM ({total_vram_gb:.1f}GB) below recommended ({min_vram}GB)")
                self.logger.warning(f"   Recommended: {model_info['recommended_gpus']}")
                self.logger.warning("   Will attempt loading with CPU offload")
            else:
                self.logger.info(f"✓ VRAM adequate: {total_vram_gb:.1f}GB available")

        # Check 2: CPU RAM
        self.logger.info("\n💾 Check 2/5: CPU RAM Availability")
        mem = psutil.virtual_memory()
        ram_gb = mem.total / (1024**3)
        available_ram_gb = mem.available / (1024**3)
        self.logger.info(f"Total RAM: {ram_gb:.1f}GB")
        self.logger.info(f"Available RAM: {available_ram_gb:.1f}GB")

        min_ram_gb = 100  # Minimum for CPU offload
        if available_ram_gb < min_ram_gb:
            self.logger.error(f"❌ FAILED: Insufficient RAM ({available_ram_gb:.1f}GB < {min_ram_gb}GB)")
            all_checks_passed = False
        else:
            self.logger.info(f"✓ PASSED: RAM adequate for CPU offload")

        # Check 3: Disk Space
        self.logger.info("\n💿 Check 3/5: Disk Space")
        disk = psutil.disk_usage('/')
        free_gb = disk.free / (1024**3)
        self.logger.info(f"Free disk space: {free_gb:.1f}GB")

        min_disk_gb = 300  # For model cache + offload
        if free_gb < min_disk_gb:
            self.logger.warning(f"⚠️  WARNING: Low disk space ({free_gb:.1f}GB < {min_disk_gb}GB recommended)")
            self.logger.warning("   May encounter issues during model download or offload")
        else:
            self.logger.info(f"✓ PASSED: Disk space adequate")

        # Check 4: Configuration File
        self.logger.info("\n⚙️  Check 4/5: Configuration File")
        if not Path(self.config_path).exists():
            self.logger.error(f"❌ FAILED: Config file not found: {self.config_path}")
            all_checks_passed = False
        else:
            self.logger.info(f"✓ PASSED: Config file found")

            # Validate config has large_llm section
            import yaml
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            if 'models' not in config or 'large_llm' not in config['models']:
                self.logger.error("❌ FAILED: Config missing 'models.large_llm' section")
                all_checks_passed = False
            else:
                large_llm_config = config['models']['large_llm']
                self.logger.info(f"  Model ID: {large_llm_config.get('model_id', 'N/A')}")
                self.logger.info(f"  4-bit quantization: {large_llm_config.get('use_4bit', False)}")
                self.logger.info(f"  Flash Attention: {large_llm_config.get('use_flash_attention', False)}")
                self.logger.info("✓ PASSED: Config valid")

        # Check 5: Dataset Access
        self.logger.info("\n📚 Check 5/5: Dataset Access")
        try:
            # Try to load a tiny subset to verify access
            dataset = load_dataset("evalplus/humanevalplus", split="test", streaming=True)
            first_item = next(iter(dataset))
            self.logger.info("✓ PASSED: HumanEval Plus dataset accessible")
        except Exception as e:
            self.logger.error(f"❌ FAILED: Cannot access dataset: {e}")
            all_checks_passed = False

        # Summary
        self.logger.info("\n" + "="*80)
        if all_checks_passed:
            self.logger.info("✅ ALL PRE-FLIGHT CHECKS PASSED")
            self.logger.info(f"Estimated loading time: {model_info['loading_time_min']} minutes")
            self.logger.info(f"Estimated total time: {self._estimate_total_time()} minutes")
        else:
            self.logger.error("❌ SOME PRE-FLIGHT CHECKS FAILED")
            self.logger.error("Cannot proceed with benchmark. Please fix issues above.")
        self.logger.info("="*80 + "\n")

        return all_checks_passed

    def _estimate_total_time(self) -> str:
        """Estimate total benchmark time"""
        model_info = self.AVAILABLE_MODELS[self.model_choice]
        loading_time_min = int(model_info["loading_time_min"].split("-")[1])  # Take upper bound

        # Estimate inference time per example
        if "405b" in self.model_choice:
            time_per_example = 90  # seconds
        elif "180b" in self.model_choice or "176b" in self.model_choice:
            time_per_example = 60
        else:  # 70B
            time_per_example = 30

        if self.skip_execution:
            time_per_example *= 0.8  # Slightly faster without execution

        total_inference_min = (self.num_examples * time_per_example) / 60
        total_min = loading_time_min + total_inference_min

        return f"{int(total_min)}-{int(total_min * 1.2)}"

    def load_humaneval_dataset(self) -> List[Dict]:
        """Load and select HumanEval Plus examples"""
        self.logger.info("\n📥 Loading HumanEval Plus dataset...")

        try:
            dataset = load_dataset("evalplus/humanevalplus", split="test")
            self.logger.info(f"Total examples in dataset: {len(dataset)}")

            # Shuffle and select
            examples = list(dataset)
            random.Random(self.seed).shuffle(examples)
            selected = examples[:self.num_examples]

            self.selected_examples = selected
            self.logger.info(f"Selected {len(selected)} examples for testing\n")

            return selected

        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise

    def initialize_large_model(self):
        """Initialize the large model system"""
        self.logger.info("\n🚀 Initializing Large Model System...")
        self.logger.info("This may take 15-25 minutes for 405B models...")
        self.logger.info("Please be patient... ☕\n")

        try:
            self.large_model_system = DirectLargeModelSystem(
                config_path=self.config_path,
                enable_carbon_tracking=True,
                logger=self.logger
            )

            self.large_model_system.initialize()
            self.logger.info("\n✅ Large model initialized successfully!\n")

        except Exception as e:
            self.logger.error(f"\n❌ Failed to initialize large model: {e}")
            raise

    def create_coding_prompt(self, problem: Dict) -> str:
        """Create prompt for code generation"""
        problem_text = problem['prompt'].strip()

        prompt = f"""You are an expert Python programmer. Complete the following function:

{problem_text}

Requirements:
- Provide only the complete function implementation
- Ensure the code is correct and handles edge cases
- Use efficient algorithms where applicable
- Do not include explanations or test cases

```python
"""
        return prompt

    def validate_code(self, code: str, problem: Dict) -> tuple:
        """
        Validate generated code (if execution not skipped)

        Returns:
            (is_correct, validation_msg, execution_time)
        """
        if not code or not code.strip():
            return False, "Empty response", 0.0

        # Clean code
        if '```python' in code:
            code = code.split('```python')[1].split('```')[0].strip()
        elif '```' in code:
            parts = code.split('```')
            if len(parts) >= 3:
                code = parts[1].strip()

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                f.write("\n\n")

                test_code = problem.get('test', '')
                if test_code:
                    f.write(test_code)
                    f.write(f"\n\ncheck({problem.get('entry_point', 'candidate')})\n")

                temp_file = f.name

            start_time = time.time()
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            exec_time = time.time() - start_time

            os.unlink(temp_file)

            if result.returncode == 0:
                return True, "All tests passed", exec_time
            else:
                error_msg = result.stderr or result.stdout or "Unknown error"
                return False, f"Test failed: {error_msg[:200]}", exec_time

        except subprocess.TimeoutExpired:
            return False, "Execution timeout (>30s)", 30.0
        except Exception as e:
            return False, f"Validation error: {str(e)}", 0.0

    def run_benchmark(self):
        """Run the complete benchmark"""
        self.logger.info("\n" + "="*80)
        self.logger.info("🏁 STARTING BENCHMARK")
        self.logger.info("="*80 + "\n")

        start_time = time.time()
        results = []

        try:
            for i, problem in enumerate(self.selected_examples, 1):
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Processing Example {i}/{len(self.selected_examples)}")
                self.logger.info(f"Task ID: {problem['task_id']}")
                self.logger.info(f"{'='*60}")

                # Create prompt
                prompt = self.create_coding_prompt(problem)

                # Generate
                gen_start = time.time()
                response_dict = self.large_model_system.generate_response(
                    prompt=prompt,
                    max_new_tokens=1024,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True
                )
                gen_time = time.time() - gen_start

                response = response_dict.get("response", "")
                model_used = response_dict.get("model_used", "unknown")

                self.logger.info(f"Generation time: {gen_time:.2f}s")
                self.logger.info(f"Response length: {len(response)} chars")

                # Validate (if not skipped)
                if self.skip_execution:
                    is_correct = None
                    validation_msg = "Skipped (--skip-execution)"
                    exec_time = 0.0
                elif response:
                    self.logger.info("Validating code...")
                    is_correct, validation_msg, exec_time = self.validate_code(response, problem)
                    self.logger.info(f"Validation: {'✓ PASSED' if is_correct else '✗ FAILED'} - {validation_msg}")
                else:
                    is_correct = False
                    validation_msg = "Generation failed"
                    exec_time = 0.0

                # Store result
                result = {
                    "example_id": i - 1,
                    "task_id": problem["task_id"],
                    "prompt_preview": prompt[:300] + "...",
                    "model_used": model_used,
                    "model_choice": self.model_choice,
                    "response": response,
                    "is_correct": is_correct,
                    "validation_message": validation_msg,
                    "generation_time": gen_time,
                    "execution_time": exec_time,
                    "total_time": gen_time + exec_time,
                    "timestamp": datetime.now().isoformat()
                }

                results.append(result)

                # Save progressively every 10 examples
                if i % 10 == 0:
                    self._save_progressive_results(results)
                    self.logger.info(f"\n💾 Progress saved ({i}/{len(self.selected_examples)} examples)")

                # Log progress
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = (len(self.selected_examples) - i) * avg_time
                self.logger.info(f"\nProgress: {i}/{len(self.selected_examples)} "
                               f"({i/len(self.selected_examples)*100:.1f}%)")
                self.logger.info(f"Elapsed: {elapsed/60:.1f} min, "
                               f"Estimated remaining: {remaining/60:.1f} min")

            # Final save
            self._save_final_results(results)

            # Print summary
            self._print_summary(results, time.time() - start_time)

        except KeyboardInterrupt:
            self.logger.warning("\n\n⚠️  Benchmark interrupted by user")
            self._save_progressive_results(results)
            self.logger.info(f"Partial results saved ({len(results)} examples)")
            raise

        except Exception as e:
            self.logger.error(f"\n❌ Benchmark failed: {e}")
            self._save_progressive_results(results)
            self.logger.info(f"Partial results saved ({len(results)} examples)")
            raise

        finally:
            if self.large_model_system:
                self.logger.info("\n🧹 Cleaning up...")
                self.large_model_system.cleanup()

    def _save_progressive_results(self, results: List[Dict]):
        """Save results progressively"""
        filename = f"{self.model_choice}_partial_{len(results)}.json"
        filepath = self.results_dir / filename

        with open(filepath, 'w') as f:
            json.dump({
                "model": self.model_choice,
                "num_examples_completed": len(results),
                "total_examples": self.num_examples,
                "skip_execution": self.skip_execution,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)

    def _save_final_results(self, results: List[Dict]):
        """Save final results"""
        filename = f"{self.model_choice}_final_{self.num_examples}.json"
        filepath = self.results_dir / filename

        # Calculate statistics
        if self.skip_execution:
            success_count = sum(1 for r in results if r["response"])
            success_rate = success_count / len(results) if results else 0
        else:
            correct_count = sum(1 for r in results if r["is_correct"])
            accuracy = correct_count / len(results) if results else 0

        avg_gen_time = sum(r["generation_time"] for r in results) / len(results) if results else 0
        avg_total_time = sum(r["total_time"] for r in results) / len(results) if results else 0

        # Get system stats
        stats = self.large_model_system.get_system_stats() if self.large_model_system else {}

        final_data = {
            "benchmark_info": {
                "model": self.model_choice,
                "model_name": self.AVAILABLE_MODELS[self.model_choice]["name"],
                "num_examples": len(results),
                "skip_execution": self.skip_execution,
                "seed": self.seed,
                "timestamp": datetime.now().isoformat()
            },
            "performance": {
                "correct_solutions": correct_count if not self.skip_execution else None,
                "accuracy": accuracy if not self.skip_execution else None,
                "success_rate": success_rate if self.skip_execution else None,
                "avg_generation_time": avg_gen_time,
                "avg_total_time": avg_total_time
            },
            "system_stats": stats,
            "results": results
        }

        with open(filepath, 'w') as f:
            json.dump(final_data, f, indent=2)

        self.logger.info(f"\n💾 Final results saved to: {filepath}")

    def _print_summary(self, results: List[Dict], total_time: float):
        """Print benchmark summary"""
        print("\n" + "="*80)
        print("📊 BENCHMARK SUMMARY")
        print("="*80)
        print(f"Model: {self.AVAILABLE_MODELS[self.model_choice]['name']} ({self.model_choice})")
        print(f"Examples: {len(results)}")
        print(f"Skip Execution: {self.skip_execution}")
        print("-"*80)

        if self.skip_execution:
            success_count = sum(1 for r in results if r["response"])
            success_rate = success_count / len(results) * 100
            print(f"Generation Success: {success_count}/{len(results)} ({success_rate:.1f}%)")
        else:
            correct_count = sum(1 for r in results if r["is_correct"])
            accuracy = correct_count / len(results) * 100
            print(f"Correct Solutions: {correct_count}/{len(results)} ({accuracy:.1f}%)")

        avg_gen_time = sum(r["generation_time"] for r in results) / len(results)
        avg_total_time = sum(r["total_time"] for r in results) / len(results)

        print(f"Avg Generation Time: {avg_gen_time:.2f}s")
        print(f"Avg Total Time: {avg_total_time:.2f}s")
        print(f"Total Benchmark Time: {total_time/60:.1f} minutes")

        # Carbon footprint
        if self.large_model_system:
            stats = self.large_model_system.get_system_stats()
            carbon = stats.get("carbon_footprint", {})
            total_co2 = carbon.get("total_emissions_kg", 0)
            print(f"Total CO2 Emissions: {total_co2:.6f} kg")

        print("="*80 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Large Model Benchmark - Test 150B+ parameter models on HumanEval Plus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test Llama-3.1-405B with 50 examples, skipping code execution
  python large_model_benchmark.py --model llama3_405b --num-examples 50 --skip-execution

  # Full test with all 164 examples and code validation
  python large_model_benchmark.py --model llama3_405b --num-examples 164

  # Test smaller 70B model (faster, good for testing)
  python large_model_benchmark.py --model codellama_70b --num-examples 10
        """
    )

    parser.add_argument("--model", choices=list(LargeModelBenchmark.AVAILABLE_MODELS.keys()),
                        required=True, help="Which large model to test")
    parser.add_argument("--config", default="configs/production_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--num-examples", type=int, default=50,
                        help="Number of examples to test (164 for full dataset)")
    parser.add_argument("--skip-execution", action="store_true",
                        help="Skip code execution (faster, only measure generation)")
    parser.add_argument("--results-dir", default="results/large_model_benchmark",
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip pre-flight checks (not recommended)")

    args = parser.parse_args()

    # Print model info
    model_info = LargeModelBenchmark.AVAILABLE_MODELS[args.model]
    print("\n" + "="*80)
    print("🚀 LARGE MODEL BENCHMARK")
    print("="*80)
    print(f"Model: {model_info['name']} ({model_info['params']})")
    print(f"Recommended Setup: {model_info['recommended_gpus']}")
    print(f"Expected Loading Time: {model_info['loading_time_min']} minutes")
    print(f"Examples to Test: {args.num_examples}")
    print(f"Code Execution: {'DISABLED' if args.skip_execution else 'ENABLED'}")
    print("="*80 + "\n")

    # Create benchmark
    benchmark = LargeModelBenchmark(
        config_path=args.config,
        model_choice=args.model,
        num_examples=args.num_examples,
        skip_execution=args.skip_execution,
        results_dir=args.results_dir,
        seed=args.seed
    )

    # Run pre-flight checks
    if not args.skip_preflight:
        if not benchmark.run_preflight_checks():
            print("\n❌ Pre-flight checks failed. Cannot proceed.")
            print("Fix the issues above or use --skip-preflight to bypass (not recommended)")
            sys.exit(1)

        input("\nPress ENTER to continue or Ctrl+C to abort...")

    # Load dataset
    benchmark.load_humaneval_dataset()

    # Initialize model
    benchmark.initialize_large_model()

    # Run benchmark
    benchmark.run_benchmark()

    print("\n✅ Benchmark completed successfully!")


if __name__ == "__main__":
    main()
