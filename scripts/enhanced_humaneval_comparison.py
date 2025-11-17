#!/usr/bin/env python3
"""
Enhanced HumanEval Plus Comparison Script
Compares RouterLLM system with dynamic MoE router vs single 100B+ parameter LLM
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
import torch
import argparse

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent))

from datasets import load_dataset
from src.routerllm.core.system import RouterLLMSystem
from src.routerllm.models.large_model_manager import DirectLargeModelSystem
from src.routerllm.models.moe_router import DynamicMoERouter
from src.routerllm.utils.logger import setup_logger

# Code execution for validation
import tempfile
import traceback


class EnhancedHumanEvalComparator:
    """
    Enhanced comparator for HumanEval Plus experiment
    Supports 100B+ models and dynamic routing
    """
    
    def __init__(
        self,
        config_path: str = "configs/production_config.yaml",
        num_examples: int = 50,
        seed: int = 42,
        results_dir: str = "results/production",
        use_dynamic_router: bool = True,
        use_large_model: bool = True,
        skip_execution: bool = False
    ):
        """
        Initialize the enhanced comparator

        Args:
            config_path: Path to production config with 100B+ models
            num_examples: Number of HumanEval examples to test
            seed: Random seed for reproducibility
            results_dir: Directory to save results
            use_dynamic_router: Whether to use dynamic MoE router
            use_large_model: Whether to use 100B+ model for comparison
            skip_execution: Whether to skip code execution (only generate, don't validate)
        """
        self.config_path = config_path
        self.num_examples = num_examples
        self.seed = seed
        self.results_dir = Path(results_dir)
        self.use_dynamic_router = use_dynamic_router
        self.use_large_model = use_large_model
        self.skip_execution = skip_execution
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logger(
            "enhanced_humaneval_comparator",
            str(self.results_dir),
            "enhanced_comparison.log"
        )
        
        # Set seeds
        random.seed(seed)
        torch.manual_seed(seed)
        
        # Initialize systems
        self.router_system = None
        self.direct_system = None
        self.dynamic_router = None
        
        # Results storage
        self.router_results = []
        self.direct_results = []
        self.selected_examples = []
        
        self.logger.info(f"Enhanced comparator initialized - Examples: {num_examples}, "
                        f"Dynamic Router: {use_dynamic_router}, Large Model: {use_large_model}, "
                        f"Skip Execution: {skip_execution}")

        if skip_execution:
            self.logger.warning("⚠️  CODE EXECUTION DISABLED - Only generating code, not validating correctness")
            self.logger.warning("This mode is faster but cannot measure actual accuracy")
        
    def load_humaneval_plus_dataset(self) -> List[Dict]:
        """Load HumanEval Plus dataset and select random examples"""
        self.logger.info("Loading HumanEval Plus dataset...")
        
        try:
            # Load the dataset
            dataset = load_dataset("evalplus/humanevalplus", split="test")
            self.logger.info(f"Loaded {len(dataset)} examples from HumanEval Plus")
            
            # Convert to list and shuffle with seed
            examples = list(dataset)
            random.Random(self.seed).shuffle(examples)
            
            # Select random examples
            selected = examples[:self.num_examples]
            self.selected_examples = selected
            
            self.logger.info(f"Selected {len(selected)} examples for testing")
            
            # Log problem categories
            categories = self._categorize_problems(selected)
            for category, count in categories.items():
                self.logger.info(f"  {category}: {count} problems")
                
            return selected
            
        except Exception as e:
            self.logger.error(f"Failed to load HumanEval Plus dataset: {e}")
            raise
            
    def _categorize_problems(self, problems: List[Dict]) -> Dict[str, int]:
        """Categorize problems by complexity"""
        categories = {
            "simple": 0,    # Basic operations, simple logic
            "medium": 0,    # Algorithms, data structures  
            "complex": 0,   # Complex algorithms, optimizations
            "very_complex": 0  # Advanced algorithms, system design
        }
        
        for problem in problems:
            prompt = problem['prompt'].lower()
            
            # Simple heuristics for categorization
            if any(word in prompt for word in ['return', 'sum', 'count', 'check', 'is_']):
                categories["simple"] += 1
            elif any(word in prompt for word in ['sort', 'search', 'find', 'filter']):
                categories["medium"] += 1
            elif any(word in prompt for word in ['dynamic', 'optimize', 'recursive', 'tree', 'graph']):
                categories["complex"] += 1
            else:
                categories["very_complex"] += 1
                
        return categories
        
    def create_coding_prompt(self, problem: Dict) -> str:
        """
        Create optimized prompt for code generation with large models
        
        Args:
            problem: HumanEval problem dictionary
            
        Returns:
            Formatted prompt string
        """
        problem_text = problem['prompt'].strip()
        
        # Generic prompt format that works well with most instruction-tuned models
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
        
    def _clean_generated_code(self, code: str) -> str:
        """Clean up generated code by removing artifacts"""
        if not code:
            return ""
            
        # Remove markdown code blocks
        if '```python' in code:
            parts = code.split('```python')
            if len(parts) > 1:
                code_part = parts[1].split('```')[0]
                code = code_part.strip()
        elif '```' in code:
            parts = code.split('```')
            if len(parts) >= 3:
                code = parts[1].strip()
                
        # Remove common prefixes that models might add
        prefixes_to_remove = [
            "Here's the solution:",
            "Here is the implementation:",
            "Solution:",
            "```python",
            "```"
        ]
        
        for prefix in prefixes_to_remove:
            if code.startswith(prefix):
                code = code[len(prefix):].strip()
                
        return code
        
    def validate_code_solution(self, code: str, problem: Dict) -> Tuple[bool, str, float]:
        """
        Validate if the generated code passes test cases
        
        Args:
            code: Generated Python code
            problem: HumanEval problem with test cases
            
        Returns:
            Tuple of (is_correct, error_message, execution_time)
        """
        if not code or not code.strip():
            return False, "Empty or blank response", 0.0
            
        # Clean the generated code
        code = self._clean_generated_code(code)
        
        try:
            # Create temporary file for testing
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Write the function
                f.write(code)
                f.write("\n\n")
                
                # Write test harness
                test_code = problem.get('test', '')
                if test_code:
                    f.write(test_code)
                    f.write(f"\n\n# Execute validation\n")
                    f.write(f"check({problem.get('entry_point', 'candidate')})\n")
                    
                temp_file = f.name
                
            # Execute with timeout
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            execution_time = time.time() - start_time
            
            # Clean up
            os.unlink(temp_file)
            
            # Check result
            if result.returncode == 0:
                return True, "All tests passed", execution_time
            else:
                error_msg = result.stderr or result.stdout or "Unknown error"
                return False, f"Test failed: {error_msg[:200]}", execution_time
                
        except subprocess.TimeoutExpired:
            return False, "Execution timeout (>30s)", 30.0
        except Exception as e:
            return False, f"Validation error: {str(e)}", 0.0
            
    def test_router_system(self) -> List[Dict]:
        """Test RouterLLM system with dynamic MoE routing"""
        if self.use_dynamic_router:
            self.logger.info("Testing RouterLLM with Dynamic MoE Router...")
            router_type = "moe_dynamic"
        else:
            self.logger.info("Testing RouterLLM with Complexity Router...")
            router_type = "graham_complexity"
            
        # Initialize RouterLLM system
        try:
            if self.use_dynamic_router:
                # Initialize dynamic MoE router
                self.dynamic_router = DynamicMoERouter(
                    encoder_model="microsoft/codebert-base",
                    num_experts=4,
                    top_k=2,
                    carbon_aware=True,
                    cost_aware=True,
                    logger=self.logger
                )
                
                # Integrate with RouterLLM system
                self.router_system = RouterLLMSystem(
                    config_path=self.config_path,
                    router_type="custom",
                    enable_carbon_tracking=True
                )
                # Replace router with dynamic MoE
                self.router_system.router = self.dynamic_router
            else:
                # Use standard complexity router
                self.router_system = RouterLLMSystem(
                    config_path=self.config_path,
                    router_type=router_type,
                    enable_carbon_tracking=True
                )
                
            self.router_system.initialize()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize router system: {e}")
            raise
            
        results = []
        
        try:
            for i, problem in enumerate(self.selected_examples):
                self.logger.info(f"Router System - Processing {i+1}/{len(self.selected_examples)}")
                
                # Create prompt
                prompt = self.create_coding_prompt(problem)
                
                # Generate response
                start_time = time.time()
                
                if self.use_dynamic_router:
                    # Use dynamic router
                    routing_decision = self.dynamic_router.forward(prompt, return_all_scores=True)
                    predicted_model = routing_decision["expert_name"]
                    confidence = routing_decision["confidence"]
                    
                    # Generate with selected model
                    self.router_system.llm_manager.load_model(predicted_model)
                    response = self.router_system.llm_manager.generate_response(
                        prompt=prompt,
                        max_new_tokens=1024,
                        temperature=0.1,
                        top_p=0.9,
                        do_sample=True
                    )
                    
                    result_dict = {
                        "response": response,
                        "predicted_model": predicted_model,
                        "confidence": confidence,
                        "routing_details": routing_decision,
                        "status": "success" if response else "error"
                    }
                else:
                    # Use standard routing
                    result_dict = self.router_system.predict_and_generate(
                        input_text=prompt,
                        max_length=1024,
                        temperature=0.1,
                        top_p=0.9,
                        do_sample=True
                    )
                    
                generation_time = time.time() - start_time

                # Validate code (skip if skip_execution flag is set)
                if self.skip_execution:
                    # Skip execution, just mark as generated
                    is_correct = None  # Unknown (not tested)
                    validation_msg = "Code generated but not executed (skip_execution=True)"
                    exec_time = 0.0
                elif result_dict["status"] == "success":
                    is_correct, validation_msg, exec_time = self.validate_code_solution(
                        result_dict["response"], problem
                    )
                else:
                    is_correct = False
                    validation_msg = result_dict.get("error", "Generation failed")
                    exec_time = 0.0
                    
                # Store result
                test_result = {
                    "example_id": i,
                    "task_id": problem["task_id"],
                    "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                    "predicted_model": result_dict.get("predicted_model", "unknown"),
                    "confidence": result_dict.get("confidence", 0.0),
                    "response": result_dict.get("response", ""),
                    "is_correct": is_correct,
                    "validation_message": validation_msg,
                    "execution_time": exec_time,
                    "generation_time": generation_time,
                    "total_time": generation_time + exec_time,
                    "status": result_dict["status"]
                }
                
                if self.use_dynamic_router and "routing_details" in result_dict:
                    test_result["estimated_carbon"] = result_dict["routing_details"].get("estimated_carbon", 0)
                    test_result["estimated_cost"] = result_dict["routing_details"].get("estimated_cost", 0)
                    
                results.append(test_result)
                
                self.logger.info(f"  Model: {test_result['predicted_model']}, "
                               f"Correct: {is_correct}, Time: {generation_time:.2f}s")
                
        finally:
            # Get system stats
            stats = self.router_system.get_system_stats()
            
            # Store results
            self.router_results = {
                "method": f"RouterLLM_{'DynamicMoE' if self.use_dynamic_router else 'ComplexityRouting'}",
                "router_type": router_type,
                "num_examples": len(results),
                "results": results,
                "system_stats": stats,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cleanup
            self.router_system.cleanup()
            
        return results
        
    def test_direct_large_model(self) -> List[Dict]:
        """Test direct 100B+ parameter LLM"""
        if self.use_large_model:
            self.logger.info("Testing direct 100B+ parameter LLM...")
            model_desc = "100B+ (Llama-3.1-405B or equivalent)"
        else:
            self.logger.info("Testing with smaller comparison model...")
            model_desc = "70B"
            
        # Initialize Direct Large Model System
        self.direct_system = DirectLargeModelSystem(
            config_path=self.config_path,
            enable_carbon_tracking=True,
            logger=self.logger
        )
        
        try:
            self.direct_system.initialize()
        except Exception as e:
            self.logger.error(f"Failed to initialize large model: {e}")
            self.logger.warning("Falling back to smaller model...")
            # Could implement fallback here
            raise
            
        results = []
        
        try:
            for i, problem in enumerate(self.selected_examples):
                self.logger.info(f"Direct LLM - Processing {i+1}/{len(self.selected_examples)}")
                
                # Use same prompt
                prompt = self.create_coding_prompt(problem)
                
                # Generate response
                start_time = time.time()
                result_dict = self.direct_system.generate_response(
                    prompt=prompt,
                    max_new_tokens=1024,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True
                )
                generation_time = time.time() - start_time

                # Validate code (skip if skip_execution flag is set)
                response = result_dict.get("response", "")
                if self.skip_execution:
                    # Skip execution, just mark as generated
                    is_correct = None  # Unknown (not tested)
                    validation_msg = "Code generated but not executed (skip_execution=True)"
                    exec_time = 0.0
                    status = "success" if response else "error"
                elif response:
                    is_correct, validation_msg, exec_time = self.validate_code_solution(
                        response, problem
                    )
                    status = "success"
                else:
                    is_correct = False
                    validation_msg = "Generation failed"
                    exec_time = 0.0
                    status = "error"
                    
                # Store result
                test_result = {
                    "example_id": i,
                    "task_id": problem["task_id"],
                    "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                    "model_used": result_dict.get("model_used", "unknown"),
                    "model_parameters": result_dict.get("model_parameters", model_desc),
                    "response": response,
                    "is_correct": is_correct,
                    "validation_message": validation_msg,
                    "execution_time": exec_time,
                    "generation_time": generation_time,
                    "total_time": generation_time + exec_time,
                    "status": status
                }
                
                results.append(test_result)
                
                self.logger.info(f"  Model: {model_desc}, "
                               f"Correct: {is_correct}, Time: {generation_time:.2f}s")
                
        finally:
            # Get system stats
            stats = self.direct_system.get_system_stats()
            
            # Store results
            self.direct_results = {
                "method": "Direct_Large_LLM",
                "model_description": model_desc,
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

        # Calculate accuracy (handle None values when skip_execution=True)
        if self.skip_execution:
            # Count successful generations instead of correct solutions
            router_correct = sum(1 for r in self.router_results["results"] if r["status"] == "success")
            direct_correct = sum(1 for r in self.direct_results["results"] if r["status"] == "success")
            accuracy_metric = "Generation Success Rate"
        else:
            router_correct = sum(1 for r in self.router_results["results"] if r["is_correct"])
            direct_correct = sum(1 for r in self.direct_results["results"] if r["is_correct"])
            accuracy_metric = "Correctness Rate"

        router_accuracy = router_correct / len(self.router_results["results"])
        direct_accuracy = direct_correct / len(self.direct_results["results"])
        
        # Calculate timing
        router_avg_gen_time = sum(r["generation_time"] for r in self.router_results["results"]) / len(self.router_results["results"])
        direct_avg_gen_time = sum(r["generation_time"] for r in self.direct_results["results"]) / len(self.direct_results["results"])
        
        router_avg_total_time = sum(r["total_time"] for r in self.router_results["results"]) / len(self.router_results["results"])
        direct_avg_total_time = sum(r["total_time"] for r in self.direct_results["results"]) / len(self.direct_results["results"])
        
        # Carbon footprint comparison
        router_carbon = self.router_results["system_stats"].get("carbon_footprint", {})
        direct_carbon = self.direct_results["system_stats"].get("carbon_footprint", {})
        
        router_total_co2 = router_carbon.get("total_emissions_kg", 0)
        direct_total_co2 = direct_carbon.get("total_emissions_kg", 0)
        
        # CO2 per correct solution (efficiency metric)
        router_co2_per_correct = router_total_co2 / max(router_correct, 1)
        direct_co2_per_correct = direct_total_co2 / max(direct_correct, 1)
        
        # Cost estimation (if using dynamic router)
        total_router_cost = 0
        total_direct_cost = 0
        
        if self.use_dynamic_router:
            for result in self.router_results["results"]:
                total_router_cost += result.get("estimated_cost", 0)
                
        # Estimate direct cost (100B+ model is expensive)
        for result in self.direct_results["results"]:
            # Rough estimate: $0.001 per 1K tokens for 100B model
            estimated_tokens = len(result["prompt"].split()) + len(result["response"].split())
            total_direct_cost += (estimated_tokens / 1000) * 0.001
            
        # Model usage analysis (for router)
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
                "router_type": "Dynamic MoE" if self.use_dynamic_router else "Complexity-based",
                "large_model_type": "100B+" if self.use_large_model else "70B",
                "skip_execution": self.skip_execution,
                "accuracy_metric": accuracy_metric,
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
                "accuracy_difference": router_accuracy - direct_accuracy,
                "relative_performance": (router_accuracy / direct_accuracy * 100) if direct_accuracy > 0 else 0
            },
            "performance_comparison": {
                "router_avg_generation_time": router_avg_gen_time,
                "direct_avg_generation_time": direct_avg_gen_time,
                "router_avg_total_time": router_avg_total_time,
                "direct_avg_total_time": direct_avg_total_time,
                "speedup_factor": direct_avg_total_time / router_avg_total_time if router_avg_total_time > 0 else 1.0
            },
            "carbon_footprint_comparison": {
                "router_system": {
                    "total_co2_kg": router_total_co2,
                    "co2_per_correct_solution": router_co2_per_correct,
                    "co2_per_example": router_total_co2 / len(self.router_results["results"]),
                    "emissions_breakdown": router_carbon.get("emissions_breakdown", {})
                },
                "direct_large_llm": {
                    "total_co2_kg": direct_total_co2,
                    "co2_per_correct_solution": direct_co2_per_correct,
                    "co2_per_example": direct_total_co2 / len(self.direct_results["results"])
                },
                "carbon_reduction_percentage": ((direct_total_co2 - router_total_co2) / direct_total_co2 * 100) if direct_total_co2 > 0 else 0,
                "efficiency_improvement": ((direct_co2_per_correct - router_co2_per_correct) / direct_co2_per_correct * 100) if direct_co2_per_correct > 0 else 0
            },
            "cost_comparison": {
                "router_total_cost": total_router_cost,
                "direct_total_cost": total_direct_cost,
                "cost_reduction_percentage": ((total_direct_cost - total_router_cost) / total_direct_cost * 100) if total_direct_cost > 0 else 0
            },
            "router_analysis": {
                "model_usage_distribution": model_usage,
                "routing_effectiveness": self._analyze_routing_effectiveness(model_usage)
            },
            "conclusion": {
                "more_accurate": "RouterLLM" if router_accuracy > direct_accuracy else "Direct LLM" if direct_accuracy > router_accuracy else "Tie",
                "more_green": "RouterLLM" if router_co2_per_correct < direct_co2_per_correct else "Direct LLM",
                "more_cost_effective": "RouterLLM" if total_router_cost < total_direct_cost else "Direct LLM",
                "overall_recommendation": self._generate_recommendation(
                    router_accuracy, direct_accuracy,
                    router_co2_per_correct, direct_co2_per_correct,
                    total_router_cost, total_direct_cost
                )
            }
        }
        
        return comparison
        
    def _analyze_routing_effectiveness(self, model_usage: Dict) -> str:
        """Analyze how effective the routing strategy was"""
        if not model_usage:
            return "No routing data available"
            
        # Calculate accuracy per model size
        model_sizes = {
            "llama3_70b": "70B",
            "codellama_34b": "34B",
            "codellama_13b": "13B",
            "deepseek_7b": "7B"
        }
        
        effectiveness = []
        for model, stats in model_usage.items():
            if stats["count"] > 0:
                accuracy = stats["correct"] / stats["count"]
                size = model_sizes.get(model, "Unknown")
                effectiveness.append(f"{model} ({size}): {stats['count']} uses, {accuracy:.1%} accuracy")
                
        return " | ".join(effectiveness)
        
    def _generate_recommendation(
        self,
        router_acc: float, direct_acc: float,
        router_co2: float, direct_co2: float,
        router_cost: float, direct_cost: float
    ) -> str:
        """Generate overall recommendation based on all metrics"""
        score = 0
        
        # Accuracy (most important)
        if router_acc >= direct_acc * 0.95:  # Within 5% is acceptable
            score += 2
        elif router_acc >= direct_acc * 0.90:  # Within 10% is OK
            score += 1
            
        # Environmental impact
        if router_co2 < direct_co2 * 0.5:  # Less than half the emissions
            score += 2
        elif router_co2 < direct_co2 * 0.75:  # 25% reduction
            score += 1
            
        # Cost
        if router_cost < direct_cost * 0.5:  # Less than half the cost
            score += 2
        elif router_cost < direct_cost * 0.75:  # 25% reduction
            score += 1
            
        if score >= 5:
            return "STRONGLY RECOMMEND RouterLLM - Excellent balance of accuracy, environmental impact, and cost"
        elif score >= 3:
            return "RECOMMEND RouterLLM - Good trade-off between performance and efficiency"
        elif score >= 2:
            return "CONDITIONAL - RouterLLM viable if efficiency is prioritized over accuracy"
        else:
            return "USE DIRECT LLM - Better accuracy outweighs efficiency gains"
            
    def save_results(self):
        """Save all results to JSON files"""
        self.logger.info("Saving results...")
        
        # Save individual results
        with open(self.results_dir / "router_results.json", "w") as f:
            json.dump(self.router_results, f, indent=2)
            
        with open(self.results_dir / "direct_results.json", "w") as f:
            json.dump(self.direct_results, f, indent=2)
            
        # Save comparison
        comparison = self.compare_results()
        with open(self.results_dir / "comparison_analysis.json", "w") as f:
            json.dump(comparison, f, indent=2)
            
        # Save selected examples
        with open(self.results_dir / "selected_examples.json", "w") as f:
            # Save without the actual code to reduce file size
            examples_summary = [
                {
                    "task_id": ex["task_id"],
                    "prompt_preview": ex["prompt"][:200] + "..." if len(ex["prompt"]) > 200 else ex["prompt"]
                }
                for ex in self.selected_examples
            ]
            json.dump(examples_summary, f, indent=2)
            
        self.logger.info(f"Results saved to {self.results_dir}")
        
        # Print summary
        self._print_summary(comparison)
        
    def _print_summary(self, comparison: Dict[str, Any]):
        """Print summary of results"""
        print("\n" + "="*80)
        print("ENHANCED HUMANEVAL COMPARISON RESULTS")
        print("="*80)
        print(f"Router Type: {comparison['experiment_info']['router_type']}")
        print(f"Large Model: {comparison['experiment_info']['large_model_type']}")
        print(f"Examples Tested: {comparison['experiment_info']['num_examples']}")
        print("-"*80)
        
        print("\nACCURACY:")
        print(f"  RouterLLM: {comparison['accuracy_comparison']['router_system']['accuracy']:.1%} "
              f"({comparison['accuracy_comparison']['router_system']['correct_solutions']}/{comparison['accuracy_comparison']['router_system']['total_examples']})")
        print(f"  Direct LLM: {comparison['accuracy_comparison']['direct_large_llm']['accuracy']:.1%} "
              f"({comparison['accuracy_comparison']['direct_large_llm']['correct_solutions']}/{comparison['accuracy_comparison']['direct_large_llm']['total_examples']})")
        print(f"  Winner: {comparison['conclusion']['more_accurate']}")
        
        print("\nPERFORMANCE:")
        print(f"  RouterLLM Avg Time: {comparison['performance_comparison']['router_avg_total_time']:.2f}s")
        print(f"  Direct LLM Avg Time: {comparison['performance_comparison']['direct_avg_total_time']:.2f}s")
        print(f"  Speedup Factor: {comparison['performance_comparison']['speedup_factor']:.2f}x")
        
        print("\nENVIRONMENTAL IMPACT:")
        print(f"  RouterLLM CO2/correct: {comparison['carbon_footprint_comparison']['router_system']['co2_per_correct_solution']:.6f} kg")
        print(f"  Direct LLM CO2/correct: {comparison['carbon_footprint_comparison']['direct_large_llm']['co2_per_correct_solution']:.6f} kg")
        print(f"  Carbon Reduction: {comparison['carbon_footprint_comparison']['carbon_reduction_percentage']:.1f}%")
        print(f"  Winner: {comparison['conclusion']['more_green']}")
        
        print("\nCOST:")
        print(f"  RouterLLM Total: ${comparison['cost_comparison']['router_total_cost']:.4f}")
        print(f"  Direct LLM Total: ${comparison['cost_comparison']['direct_total_cost']:.4f}")
        print(f"  Cost Reduction: {comparison['cost_comparison']['cost_reduction_percentage']:.1f}%")
        print(f"  Winner: {comparison['conclusion']['more_cost_effective']}")
        
        print("\nROUTER EFFECTIVENESS:")
        print(f"  {comparison['router_analysis']['routing_effectiveness']}")
        
        print("\nOVERALL RECOMMENDATION:")
        print(f"  {comparison['conclusion']['overall_recommendation']}")
        print("="*80)
        
    def run_comparison(self):
        """Run the complete comparison experiment"""
        self.logger.info("Starting Enhanced HumanEval Plus comparison...")
        
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
    parser = argparse.ArgumentParser(description="Enhanced HumanEval Plus Comparison")
    parser.add_argument("--config", default="configs/production_config.yaml", help="Config file path")
    parser.add_argument("--num-examples", type=int, default=50, help="Number of examples to test (use 164 for full dataset)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--results-dir", default="results/production", help="Results directory")
    parser.add_argument("--use-dynamic-router", action="store_true", help="Use dynamic MoE router")
    parser.add_argument("--use-large-model", action="store_true", help="Use 100B+ parameter model")
    parser.add_argument("--skip-execution", action="store_true",
                        help="Skip code execution validation (faster, only generate code without testing)")
    parser.add_argument("--gpu-memory", type=str, help="Override GPU memory limit (e.g., '80GB')")

    args = parser.parse_args()

    if args.skip_execution:
        print("⚠️  WARNING: Code execution is DISABLED. Only code generation will be tested.")
        print("    Accuracy metrics will show generation success rate, not correctness.")
        print()
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}, Memory: {props.total_memory / 1024**3:.1f}GB")
    else:
        print("WARNING: No GPUs available. Large models may not load properly.")
        
    # Create and run comparator
    comparator = EnhancedHumanEvalComparator(
        config_path=args.config,
        num_examples=args.num_examples,
        seed=args.seed,
        results_dir=args.results_dir,
        use_dynamic_router=args.use_dynamic_router,
        use_large_model=args.use_large_model,
        skip_execution=args.skip_execution
    )

    comparator.run_comparison()


if __name__ == "__main__":
    main()
