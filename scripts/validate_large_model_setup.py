#!/usr/bin/env python3
"""
Validate Large Model Setup
Quick validation script to test if your hardware can load and run large models
Run this before attempting the full benchmark to avoid wasting time
"""

import sys
import time
import torch
import psutil
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent))

from src.routerllm.models.large_model_manager import DirectLargeModelSystem
from src.routerllm.utils.logger import setup_logger


class LargeModelValidator:
    """Quick validator for large model setup"""

    TEST_PROMPT = """You are an expert Python programmer. Complete the following function:

def add_two_numbers(a: int, b: int) -> int:
    \"\"\"Add two numbers together.\"\"\"

Requirements:
- Provide only the complete function implementation

```python
"""

    def __init__(self, config_path: str):
        """
        Initialize validator

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.logger = setup_logger(
            "large_model_validator",
            "./logs",
            f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

    def print_system_info(self):
        """Print current system information"""
        print("\n" + "="*80)
        print("🖥️  SYSTEM INFORMATION")
        print("="*80)

        # GPU Info
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"\n📊 GPUs: {num_gpus} device(s) detected")

            total_vram = 0
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024**3)
                total_vram += vram_gb

                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                free = vram_gb - reserved

                print(f"  GPU {i}: {props.name}")
                print(f"    Total: {vram_gb:.1f}GB")
                print(f"    Allocated: {allocated:.1f}GB")
                print(f"    Reserved: {reserved:.1f}GB")
                print(f"    Free: {free:.1f}GB")

            print(f"\n  Total VRAM: {total_vram:.1f}GB")
        else:
            print("\n❌ No CUDA GPUs detected!")
            print("Large models (100B+) require GPU acceleration")
            return False

        # CPU/RAM Info
        mem = psutil.virtual_memory()
        ram_total = mem.total / (1024**3)
        ram_available = mem.available / (1024**3)
        ram_used = (mem.total - mem.available) / (1024**3)
        ram_percent = mem.percent

        print(f"\n💾 RAM:")
        print(f"  Total: {ram_total:.1f}GB")
        print(f"  Used: {ram_used:.1f}GB ({ram_percent:.1f}%)")
        print(f"  Available: {ram_available:.1f}GB")

        # Disk Info
        disk = psutil.disk_usage('/')
        disk_total = disk.total / (1024**3)
        disk_used = disk.used / (1024**3)
        disk_free = disk.free / (1024**3)
        disk_percent = disk.percent

        print(f"\n💿 Disk:")
        print(f"  Total: {disk_total:.1f}GB")
        print(f"  Used: {disk_used:.1f}GB ({disk_percent:.1f}%)")
        print(f"  Free: {disk_free:.1f}GB")

        print("="*80 + "\n")

        return True

    def validate_model_loading(self) -> bool:
        """
        Attempt to load the large model and test generation

        Returns:
            True if successful, False otherwise
        """
        print("\n" + "="*80)
        print("🔄 TESTING MODEL LOADING")
        print("="*80)
        print("\nThis will take 10-25 minutes depending on model size...")
        print("Please be patient... ☕\n")

        try:
            # Initialize system
            print("⏳ Step 1/3: Initializing Large Model System...")
            start_time = time.time()

            system = DirectLargeModelSystem(
                config_path=self.config_path,
                enable_carbon_tracking=False,  # Faster for validation
                logger=self.logger
            )

            # Load model
            print("⏳ Step 2/3: Loading large model...")
            print("(This is the time-consuming step)\n")
            load_start = time.time()

            system.initialize()

            load_time = time.time() - load_start
            print(f"\n✅ Model loaded successfully in {load_time/60:.1f} minutes!")

            # Print memory usage after loading
            self._print_memory_usage("AFTER MODEL LOADING")

            # Test generation
            print("\n⏳ Step 3/3: Testing generation with simple prompt...")
            gen_start = time.time()

            response = system.generate_response(
                prompt=self.TEST_PROMPT,
                max_new_tokens=256,
                temperature=0.1
            )

            gen_time = time.time() - gen_start
            print(f"\n✅ Generation completed in {gen_time:.2f} seconds!")

            if response and response.get("response"):
                print(f"\nGenerated code ({len(response['response'])} chars):")
                print("-" * 60)
                print(response["response"][:500])
                if len(response["response"]) > 500:
                    print("...")
                print("-" * 60)
            else:
                print("\n⚠️  WARNING: Generation returned empty response")
                return False

            # Print final memory usage
            self._print_memory_usage("AFTER GENERATION")

            # Cleanup
            print("\n🧹 Cleaning up...")
            system.cleanup()

            total_time = time.time() - start_time

            # Print summary
            print("\n" + "="*80)
            print("✅ VALIDATION SUCCESSFUL!")
            print("="*80)
            print(f"Model Loading Time: {load_time/60:.1f} minutes")
            print(f"Generation Time: {gen_time:.2f} seconds")
            print(f"Total Test Time: {total_time/60:.1f} minutes")
            print("\n📊 Benchmark Estimates:")
            print(f"  - 50 examples: ~{(load_time + 50 * gen_time) / 60:.0f} minutes")
            print(f"  - 164 examples: ~{(load_time + 164 * gen_time) / 60:.0f} minutes")
            print("="*80 + "\n")

            return True

        except torch.cuda.OutOfMemoryError as e:
            print("\n" + "="*80)
            print("❌ OUT OF MEMORY ERROR")
            print("="*80)
            print(f"Error: {e}")
            print("\n💡 SOLUTIONS:")
            print("1. ✓ Ensure 4-bit quantization is enabled in production_config.yaml")
            print("2. ✓ Close other GPU-using processes")
            print("3. ✓ Try a smaller model (e.g., 70B instead of 405B)")
            print("4. ✓ Use more GPUs if available")
            print("5. ✓ Increase CPU offload memory in config")
            print("="*80 + "\n")
            return False

        except Exception as e:
            print("\n" + "="*80)
            print("❌ VALIDATION FAILED")
            print("="*80)
            print(f"Error: {e}")
            print(f"Error type: {type(e).__name__}")

            import traceback
            print("\nFull traceback:")
            print(traceback.format_exc())

            print("\n💡 TROUBLESHOOTING:")
            print("1. Check config file exists and is valid")
            print("2. Verify model_id in config is correct")
            print("3. Ensure internet connection for model download")
            print("4. Check HuggingFace token if model requires authentication")
            print("5. Verify CUDA and GPU drivers are installed")
            print("="*80 + "\n")
            return False

    def _print_memory_usage(self, title: str):
        """Print current memory usage"""
        print(f"\n📊 MEMORY USAGE {title}")
        print("-" * 60)

        # CPU Memory
        mem = psutil.virtual_memory()
        ram_used = (mem.total - mem.available) / (1024**3)
        ram_percent = mem.percent
        print(f"CPU RAM: {ram_used:.1f}GB ({ram_percent:.1f}%)")

        # GPU Memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                percent = (reserved / total) * 100
                print(f"GPU {i}: {reserved:.1f}GB / {total:.1f}GB ({percent:.1f}%)")

        print("-" * 60)

    def run_validation(self) -> bool:
        """Run complete validation"""
        print("\n" + "="*80)
        print("🔍 LARGE MODEL SETUP VALIDATOR")
        print("="*80)
        print("\nThis script will:")
        print("1. Check your system resources (GPU, RAM, disk)")
        print("2. Attempt to load the large model from config")
        print("3. Test a simple code generation")
        print("4. Report time estimates for full benchmark")
        print("\n⚠️  Note: This will take 10-25 minutes depending on model size")
        print("="*80)

        # Step 1: Print system info
        if not self.print_system_info():
            return False

        # Step 2: Validate model loading
        success = self.validate_model_loading()

        if success:
            print("\n✅ Your setup is READY for large model benchmarks!")
            print("You can now run the full benchmark with confidence.\n")
        else:
            print("\n❌ Your setup has ISSUES that need to be fixed")
            print("Please address the errors above before running the benchmark.\n")

        return success


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Validate Large Model Setup - Test if your hardware can run large models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script quickly validates that your system can load and run large models
before attempting a full benchmark. It will:

1. Display your system's GPU, RAM, and disk resources
2. Load the large model specified in your config
3. Test a simple code generation
4. Estimate benchmark times

Run this BEFORE running the full benchmark to catch issues early!

Example:
  python validate_large_model_setup.py --config configs/production_config.yaml
        """
    )

    parser.add_argument("--config", default="configs/production_config.yaml",
                        help="Path to configuration file with large_llm settings")

    args = parser.parse_args()

    # Check config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"\n❌ ERROR: Config file not found: {args.config}")
        print("Please specify a valid config file with --config\n")
        sys.exit(1)

    # Run validation
    validator = LargeModelValidator(str(config_path))

    try:
        success = validator.run_validation()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Validation interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
