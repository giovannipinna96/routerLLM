#!/usr/bin/env python3
"""
Aggregate Router Results Script

This script reads individual router test results (JSON files) and aggregates
them into a single CSV file for easy comparison and analysis.

Usage:
    python aggregate_results.py --results-dir ./results --output-csv ./results/router_comparison.csv
"""

import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict
import sys


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Aggregate RouterLLM test results into CSV"
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Directory containing JSON result files'
    )

    parser.add_argument(
        '--output-csv',
        type=str,
        required=True,
        help='Output CSV file path'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information'
    )

    return parser.parse_args()


def load_json_results(results_dir: Path, verbose: bool = False) -> List[Dict]:
    """
    Load all JSON result files from directory

    Args:
        results_dir: Path to results directory
        verbose: Print verbose output

    Returns:
        List of result dictionaries
    """
    results = []

    json_files = list(results_dir.glob("*_results.json"))

    if verbose:
        print(f"Found {len(json_files)} JSON result files")

    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append(data)

            if verbose:
                print(f"  Loaded: {json_file.name}")

        except Exception as e:
            print(f"  ERROR loading {json_file.name}: {e}", file=sys.stderr)

    return results


def aggregate_to_csv(results: List[Dict], output_csv: Path, verbose: bool = False):
    """
    Aggregate results into CSV format

    Args:
        results: List of result dictionaries
        output_csv: Path to output CSV file
        verbose: Print verbose output
    """
    if not results:
        print("WARNING: No results to aggregate", file=sys.stderr)
        return

    # Prepare CSV rows
    csv_rows = []

    for result in results:
        try:
            # Extract basic info
            router_name = result.get("router_name", "Unknown")
            router_type = result.get("router_type", "Unknown")
            router_model = result.get("router_model", "N/A")

            # Extract summary stats
            summary = result.get("summary_stats", {})
            total_prompts = summary.get("total_prompts", 0)
            successful = summary.get("successful", 0)
            failed = summary.get("failed", 0)
            success_rate = summary.get("success_rate", 0)
            avg_total_time = summary.get("average_total_time", 0)
            avg_router_time = summary.get("average_router_time", 0)
            avg_generation_time = summary.get("average_generation_time", 0)
            most_selected_model = summary.get("most_selected_model", "N/A")

            # Extract model selection distribution
            model_selections = summary.get("model_selections", {})
            model_selection_str = "; ".join([f"{k}:{v}" for k, v in model_selections.items()])

            # Extract carbon footprint
            carbon = result.get("carbon_footprint", {})
            total_emissions = carbon.get("total_emissions_kg", 0)
            avg_emissions_per_request = carbon.get("average_emissions_per_request", 0)

            # Extract emissions breakdown
            breakdown = carbon.get("emissions_breakdown", {})
            router_emissions = breakdown.get("router_inference", 0)
            loading_emissions = breakdown.get("model_loading", 0)
            llm_emissions = breakdown.get("llm_inference", 0)

            # Create CSV row
            row = {
                "Router Name": router_name,
                "Router Type": router_type,
                "Router Model": router_model or "N/A",
                "Total Prompts": total_prompts,
                "Successful": successful,
                "Failed": failed,
                "Success Rate (%)": f"{success_rate * 100:.2f}",
                "Avg Total Time (s)": f"{avg_total_time:.4f}",
                "Avg Router Time (s)": f"{avg_router_time:.4f}",
                "Avg Generation Time (s)": f"{avg_generation_time:.4f}",
                "Most Selected Model": most_selected_model,
                "Model Selections": model_selection_str,
                "Total CO2 Emissions (kg)": f"{total_emissions:.6f}",
                "Avg CO2 per Request (kg)": f"{avg_emissions_per_request:.6f}",
                "Router CO2 (kg)": f"{router_emissions:.6f}",
                "Model Loading CO2 (kg)": f"{loading_emissions:.6f}",
                "LLM Inference CO2 (kg)": f"{llm_emissions:.6f}"
            }

            csv_rows.append(row)

        except Exception as e:
            print(f"ERROR processing result for {result.get('router_name', 'Unknown')}: {e}", file=sys.stderr)

    # Write to CSV
    if csv_rows:
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = list(csv_rows[0].keys())

        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        if verbose:
            print(f"\nCSV file created: {output_csv}")
            print(f"Total rows: {len(csv_rows)}")

    else:
        print("ERROR: No CSV rows generated", file=sys.stderr)


def print_summary(results: List[Dict]):
    """
    Print summary of aggregated results

    Args:
        results: List of result dictionaries
    """
    print(f"\n{'='*60}")
    print("RouterLLM Results Summary")
    print(f"{'='*60}\n")

    if not results:
        print("No results to summarize")
        return

    # Sort by total emissions for ranking
    sorted_results = sorted(
        results,
        key=lambda x: x.get("carbon_footprint", {}).get("total_emissions_kg", float('inf'))
    )

    print(f"{'Router':<25} {'Success Rate':<15} {'Avg Time (s)':<15} {'CO2 (kg)':<15}")
    print(f"{'-'*70}")

    for result in sorted_results:
        router_name = result.get("router_name", "Unknown")
        summary = result.get("summary_stats", {})
        carbon = result.get("carbon_footprint", {})

        success_rate = summary.get("success_rate", 0) * 100
        avg_time = summary.get("average_total_time", 0)
        total_co2 = carbon.get("total_emissions_kg", 0)

        print(f"{router_name:<25} {success_rate:>6.1f}%        {avg_time:>6.3f}         {total_co2:>6.6f}")

    print(f"\n{'='*60}\n")

    # Highlight best performers
    if len(sorted_results) >= 2:
        print("Highlights:")
        print(f"  Lowest CO2 emissions: {sorted_results[0].get('router_name', 'Unknown')}")

        # Find fastest
        fastest = min(
            sorted_results,
            key=lambda x: x.get("summary_stats", {}).get("average_total_time", float('inf'))
        )
        print(f"  Fastest average time: {fastest.get('router_name', 'Unknown')}")

        # Find highest success rate
        best_success = max(
            sorted_results,
            key=lambda x: x.get("summary_stats", {}).get("success_rate", 0)
        )
        print(f"  Highest success rate: {best_success.get('router_name', 'Unknown')}")

    print(f"\n{'='*60}\n")


def main():
    """Main entry point"""
    args = parse_args()

    print(f"\n{'='*60}")
    print("RouterLLM Results Aggregation")
    print(f"{'='*60}")
    print(f"Results directory: {args.results_dir}")
    print(f"Output CSV: {args.output_csv}")
    print(f"{'='*60}\n")

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Load JSON results
    print("Loading JSON result files...")
    results = load_json_results(results_dir, verbose=args.verbose)

    if not results:
        print("ERROR: No valid result files found", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(results)} result files\n")

    # Aggregate to CSV
    print("Aggregating results to CSV...")
    output_csv = Path(args.output_csv)
    aggregate_to_csv(results, output_csv, verbose=args.verbose)

    # Print summary
    print_summary(results)

    print(f"Aggregation completed successfully!")
    print(f"CSV file: {output_csv}\n")


if __name__ == "__main__":
    main()
