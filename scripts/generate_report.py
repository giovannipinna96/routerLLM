#!/usr/bin/env python3
"""
Generate Final Analysis Report for RouterLLM vs Direct LLM Comparison
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class ReportGenerator:
    """Generate comprehensive analysis reports"""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)

    def load_results(self) -> Dict[str, Any]:
        """Load all result files"""
        results = {}

        # Load comparison analysis
        comparison_file = self.results_dir / "comparison_analysis.json"
        if comparison_file.exists():
            with open(comparison_file) as f:
                results["comparison"] = json.load(f)

        # Load router results
        router_file = self.results_dir / "router_results.json"
        if router_file.exists():
            with open(router_file) as f:
                results["router"] = json.load(f)

        # Load direct results
        direct_file = self.results_dir / "direct_results.json"
        if direct_file.exists():
            with open(direct_file) as f:
                results["direct"] = json.load(f)

        # Load selected examples
        examples_file = self.results_dir / "selected_examples.json"
        if examples_file.exists():
            with open(examples_file) as f:
                results["examples"] = json.load(f)

        return results

    def generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive markdown report"""

        if "comparison" not in results:
            return "# Error: No comparison data found"

        comp = results["comparison"]

        report = f"""# RouterLLM vs Large LLM Comparison Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Experiment Date**: {comp['experiment_info']['timestamp'][:10]}

## Executive Summary

This report compares the RouterLLM system (with complexity-based routing) against a single large language model on coding tasks from the HumanEval Plus dataset.

### Key Findings

- **RouterLLM Accuracy**: {comp['accuracy_comparison']['router_system']['accuracy']:.1%} ({comp['accuracy_comparison']['router_system']['correct_solutions']}/{comp['accuracy_comparison']['router_system']['total_examples']} correct)
- **Direct LLM Accuracy**: {comp['accuracy_comparison']['direct_large_llm']['accuracy']:.1%} ({comp['accuracy_comparison']['direct_large_llm']['correct_solutions']}/{comp['accuracy_comparison']['direct_large_llm']['total_examples']} correct)
- **Winner (Accuracy)**: {comp['conclusion']['more_accurate']}
- **Winner (Environmental)**: {comp['conclusion']['more_green']}

---

## Methodology

### Dataset
- **Source**: HumanEval Plus (evalplus/humanevalplus)
- **Examples Tested**: {comp['experiment_info']['num_examples']}
- **Selection**: Random sampling with seed {comp['experiment_info']['seed']}
- **Task Type**: Python coding problems with test case validation

### Systems Compared

#### RouterLLM System
"""

        # Add router information if available
        if "router" in results:
            router_info = results["router"]
            report += f"""- **Router Type**: {router_info.get('router_type', 'Unknown')}
- **Model Selection**: Based on complexity classification
- **Models Used**:
"""
            # Add model usage distribution
            model_usage = comp.get('router_analysis', {}).get('model_usage_distribution', {})
            for model, stats in model_usage.items():
                accuracy = (stats['correct'] / stats['count'] * 100) if stats['count'] > 0 else 0
                report += f"  - {model}: {stats['count']} requests, {stats['correct']} correct ({accuracy:.1f}%)\n"

        # Add direct LLM info
        if "direct" in results:
            direct_info = results["direct"]
            report += f"""
#### Direct Large LLM
- **Model**: {direct_info.get('model_used', 'Unknown')}
- **Approach**: Single large model for all tasks
- **Parameters**: 13B (CodeLlama-13B-Instruct)
"""

        # Performance Analysis
        report += f"""
---

## Performance Analysis

### Accuracy Comparison

| System | Correct Solutions | Total | Accuracy | Difference |
|--------|------------------|-------|----------|------------|
| RouterLLM | {comp['accuracy_comparison']['router_system']['correct_solutions']} | {comp['accuracy_comparison']['router_system']['total_examples']} | {comp['accuracy_comparison']['router_system']['accuracy']:.1%} | {comp['accuracy_comparison']['accuracy_difference']:+.1%} |
| Direct LLM | {comp['accuracy_comparison']['direct_large_llm']['correct_solutions']} | {comp['accuracy_comparison']['direct_large_llm']['total_examples']} | {comp['accuracy_comparison']['direct_large_llm']['accuracy']:.1%} | - |

### Timing Analysis

| System | Average Time | Total Time |
|--------|-------------|------------|
| RouterLLM | {comp['performance_comparison']['router_avg_time_seconds']:.2f}s | {comp['performance_comparison']['router_avg_time_seconds'] * comp['experiment_info']['num_examples']:.2f}s |
| Direct LLM | {comp['performance_comparison']['direct_avg_time_seconds']:.2f}s | {comp['performance_comparison']['direct_avg_time_seconds'] * comp['experiment_info']['num_examples']:.2f}s |

**Time Difference**: RouterLLM is {comp['performance_comparison']['time_difference']:.2f}s slower per request on average.

---

## Environmental Impact Analysis

### Carbon Footprint Comparison

| System | Total CO2 (kg) | CO2 per Correct Solution (kg) | Efficiency |
|--------|---------------|------------------------------|-------------|
| RouterLLM | {comp['carbon_footprint_comparison']['router_system']['total_co2_kg']:.6f} | {comp['carbon_footprint_comparison']['router_system']['co2_per_correct_solution']:.6f} | {comp['carbon_footprint_comparison']['co2_efficiency_improvement']:.1%} better than direct |
| Direct LLM | {comp['carbon_footprint_comparison']['direct_large_llm']['total_co2_kg']:.6f} | {comp['carbon_footprint_comparison']['direct_large_llm']['co2_per_correct_solution']:.6f} | Baseline |

### RouterLLM Carbon Breakdown
"""

        # Add carbon breakdown
        breakdown = comp['carbon_footprint_comparison']['router_system'].get('emissions_breakdown', {})
        if breakdown:
            total_co2 = comp['carbon_footprint_comparison']['router_system']['total_co2_kg']
            for component, co2 in breakdown.items():
                percentage = (co2 / total_co2 * 100) if total_co2 > 0 else 0
                report += f"- **{component.replace('_', ' ').title()}**: {co2:.6f} kg ({percentage:.1f}%)\n"

        # Conclusions
        report += f"""
---

## Conclusions

### Which System Produces More Correct Code?
**Winner: {comp['conclusion']['more_accurate']}**

"""

        if comp['conclusion']['more_accurate'] == 'RouterLLM':
            report += f"""The RouterLLM system achieved higher accuracy ({comp['accuracy_comparison']['router_system']['accuracy']:.1%} vs {comp['accuracy_comparison']['direct_large_llm']['accuracy']:.1%}), demonstrating that intelligent routing can improve performance by selecting appropriate models for different complexity levels."""
        elif comp['conclusion']['more_accurate'] == 'Direct LLM':
            report += f"""The direct large LLM achieved higher accuracy ({comp['accuracy_comparison']['direct_large_llm']['accuracy']:.1%} vs {comp['accuracy_comparison']['router_system']['accuracy']:.1%}), suggesting that the consistent use of a large model may be more reliable for coding tasks."""
        else:
            report += "Both systems achieved equal accuracy, indicating comparable performance."

        report += f"""

### Which System is More Green/Ecological?
**Winner: {comp['conclusion']['more_green']}**

"""

        router_co2_per_correct = comp['carbon_footprint_comparison']['router_system']['co2_per_correct_solution']
        direct_co2_per_correct = comp['carbon_footprint_comparison']['direct_large_llm']['co2_per_correct_solution']

        if comp['conclusion']['more_green'] == 'RouterLLM':
            if direct_co2_per_correct > 0:
                improvement = (direct_co2_per_correct - router_co2_per_correct) / direct_co2_per_correct * 100
                report += f"""The RouterLLM system is more environmentally friendly, using {improvement:.1f}% less CO2 per correct solution ({router_co2_per_correct:.6f} kg vs {direct_co2_per_correct:.6f} kg). The intelligent routing reduces computational overhead by using smaller models when appropriate."""
            else:
                report += f"""The RouterLLM system produces measurable CO2 emissions ({router_co2_per_correct:.6f} kg per correct solution) while the direct system showed no measured emissions in this test."""
        else:
            if router_co2_per_correct > 0 and direct_co2_per_correct >= 0:
                overhead = (router_co2_per_correct - direct_co2_per_correct) / max(direct_co2_per_correct, 0.000001) * 100
                report += f"""The direct LLM system is more environmentally friendly, though this may be due to measurement limitations in the test setup. The RouterLLM system shows {overhead:.1f}% higher emissions per correct solution."""

        report += f"""

### Trade-off Analysis

The comparison reveals important trade-offs:

1. **Accuracy vs Efficiency**: RouterLLM may sacrifice some consistency for the potential of using more efficient models
2. **Complexity vs Simplicity**: The routing overhead adds system complexity but may provide better resource utilization
3. **Environmental vs Performance**: Carbon measurement showed {comp['conclusion']['more_green']} to be more environmentally friendly

### Recommendations

Based on this analysis:
"""

        # Add specific recommendations based on results
        if comp['conclusion']['more_accurate'] == 'RouterLLM' and comp['conclusion']['more_green'] == 'RouterLLM':
            report += """
- **Recommend RouterLLM**: Shows superior accuracy and environmental performance
- **Use Case**: Production systems where both quality and sustainability matter
- **Investment**: Worth the additional routing complexity
"""
        elif comp['conclusion']['more_accurate'] == 'Direct LLM' and comp['conclusion']['more_green'] == 'Direct LLM':
            report += """
- **Recommend Direct LLM**: Superior in both accuracy and environmental impact
- **Use Case**: When simplicity and consistency are priorities
- **Consideration**: RouterLLM may improve with better complexity classification
"""
        else:
            report += """
- **Mixed Results**: Each system has advantages in different dimensions
- **Context-Dependent**: Choice depends on whether accuracy or environmental impact is prioritized
- **Further Testing**: Consider expanding evaluation to more examples and diverse tasks
"""

        report += f"""

---

## Technical Details

### Model Configuration
- **Large Model**: CodeLlama-13B-Instruct (13B parameters)
- **Medium Models**: Mistral-7B, CodeLlama-7B (7B parameters)
- **Small Model**: Phi-3-Mini (3.8B parameters)
- **Quantization**: 4-bit quantization enabled for large models
- **GPU**: NVIDIA A100 80GB PCIe

### Limitations
- Limited to {comp['experiment_info']['num_examples']} examples (statistical significance limited)
- Complexity router may need fine-tuning for coding tasks specifically
- Carbon measurement accuracy depends on hardware monitoring capabilities
- Code validation based on test case execution (may not catch all edge cases)

### Future Work
- Expand evaluation to larger sample sizes
- Test with different complexity classifiers (BERT-based custom router)
- Evaluate on diverse programming languages and task types
- Implement more sophisticated model selection strategies
- Compare against other intelligent routing approaches

---

*Report generated by RouterLLM Analysis System*
*For questions or technical details, refer to the source code and raw result files*
"""

        return report

    def save_report(self, report: str, filename: str = "analysis_report.md"):
        """Save report to file"""
        output_path = self.results_dir / filename
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")

    def generate_json_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate JSON summary for programmatic access"""
        if "comparison" not in results:
            return {"error": "No comparison data found"}

        comp = results["comparison"]

        summary = {
            "experiment": {
                "date": comp['experiment_info']['timestamp'][:10],
                "examples_tested": comp['experiment_info']['num_examples'],
                "seed": comp['experiment_info']['seed']
            },
            "results": {
                "accuracy": {
                    "router_llm": comp['accuracy_comparison']['router_system']['accuracy'],
                    "direct_llm": comp['accuracy_comparison']['direct_large_llm']['accuracy'],
                    "winner": comp['conclusion']['more_accurate']
                },
                "environmental": {
                    "router_co2_per_correct": comp['carbon_footprint_comparison']['router_system']['co2_per_correct_solution'],
                    "direct_co2_per_correct": comp['carbon_footprint_comparison']['direct_large_llm']['co2_per_correct_solution'],
                    "winner": comp['conclusion']['more_green']
                },
                "performance": {
                    "router_avg_time": comp['performance_comparison']['router_avg_time_seconds'],
                    "direct_avg_time": comp['performance_comparison']['direct_avg_time_seconds'],
                    "time_difference": comp['performance_comparison']['time_difference']
                }
            },
            "conclusions": {
                "more_accurate_system": comp['conclusion']['more_accurate'],
                "more_green_system": comp['conclusion']['more_green'],
                "recommendation": "RouterLLM" if (comp['conclusion']['more_accurate'] == 'RouterLLM' and comp['conclusion']['more_green'] == 'RouterLLM') else "Mixed - depends on priorities"
            }
        }

        return summary


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate analysis report")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--output", default="analysis_report.md", help="Output filename")
    parser.add_argument("--json", action="store_true", help="Also generate JSON summary")

    args = parser.parse_args()

    # Generate report
    generator = ReportGenerator(args.results_dir)
    results = generator.load_results()

    if not results:
        print("Error: No results found in", args.results_dir)
        return

    # Generate markdown report
    report = generator.generate_markdown_report(results)
    generator.save_report(report, args.output)

    # Generate JSON summary if requested
    if args.json:
        summary = generator.generate_json_summary(results)
        json_path = Path(args.results_dir) / "analysis_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"JSON summary saved to: {json_path}")

    print("Report generation completed!")


if __name__ == "__main__":
    main()