# Large Model Setup Guide (150B+ Parameters)

Complete guide for running 150B-405B parameter models on RouterLLM with limited GPU resources (2-3 GPUs).

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Configuration](#configuration)
- [Accelerate Setup](#accelerate-setup)
- [Running Validation](#running-validation)
- [Running Benchmarks](#running-benchmarks)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)
- [Best Practices](#best-practices)

---

## Hardware Requirements

### Minimum Requirements (with 4-bit quantization)

| Model | GPUs | VRAM | CPU RAM | Disk Space |
|-------|------|------|---------|------------|
| **Llama-3.1-405B** | 2-3 x A100/H100 | 160-240GB | 200GB | 300GB |
| **Falcon-180B** | 2 x A100/H100 | 140-180GB | 150GB | 250GB |
| **BLOOM-176B** | 2 x A100/H100 | 140-180GB | 150GB | 250GB |
| **CodeLlama-70B** | 1-2 x A100 | 80-120GB | 100GB | 200GB |

### Recommended Setup

- **GPUs**: 3-4 x NVIDIA A100 80GB or H100 80GB with NVLink
- **CPU RAM**: 256GB+ for comfortable CPU offload
- **Disk**: 500GB+ fast SSD (NVMe preferred) for model cache and offload
- **Network**: High-bandwidth connection for initial model download (~800GB for 405B)

### Notes

- **4-bit quantization is MANDATORY** for 405B models with 2-3 GPUs
- CPU offload is automatically enabled when GPU memory is insufficient
- Disk offload is used during model loading to reduce peak memory usage

---

## Software Requirements

### Python Environment

```bash
# Required Python version
Python 3.10+

# Core dependencies
torch >= 2.0.0
transformers >= 4.35.0
accelerate >= 0.25.0
bitsandbytes >= 0.41.0

# Optional (but recommended)
flash-attn >= 2.3.0  # For Flash Attention 2
```

### Install Dependencies

```bash
# Using UV (recommended)
uv pip install torch transformers accelerate bitsandbytes

# Optional: Flash Attention 2 (requires CUDA 11.8+)
uv pip install flash-attn --no-build-isolation

# Or using pip
pip install torch transformers accelerate bitsandbytes flash-attn
```

### CUDA Version

- **Required**: CUDA 11.8+ or CUDA 12.0+
- Check with: `nvidia-smi`
- Flash Attention 2 requires CUDA 11.8+

---

## Configuration

### Production Config for 2-3 GPUs

Edit `configs/production_config.yaml`:

```yaml
models:
  large_llm:
    name: "llama3_405b"
    model_id: "meta-llama/Llama-3.1-405B-Instruct"

    # Memory allocation for 2-3 GPU setup
    max_memory:
      0: "70GB"    # GPU 0
      1: "70GB"    # GPU 1
      2: "70GB"    # GPU 2 (if available)
      cpu: "200GB" # CPU offload

    # MANDATORY for 405B with limited GPUs
    use_4bit: true
    use_flash_attention: true
    device_map: "auto"

    # Offload configuration
    offload_folder: "./offload"
    offload_state_dict: true
```

### Alternative Models (Easier to Run)

```yaml
# For testing or smaller setups
models:
  large_llm:
    name: "codellama_70b"
    model_id: "codellama/CodeLlama-70b-Instruct-hf"
    max_memory:
      0: "40GB"
      1: "40GB"
    use_4bit: true
    # Much faster: ~5 min loading, ~30 sec/generation
```

---

## Accelerate Setup

### How Accelerate Works

The updated `LargeModelManager` uses Accelerate's advanced features:

1. **`init_empty_weights()`**: Initializes model structure without loading weights
2. **`infer_auto_device_map()`**: Computes optimal layer distribution across devices
3. **`load_checkpoint_and_dispatch()`**: Loads weights with computed device map
4. **`no_split_module_classes`**: Prevents splitting attention layers (maintains performance)

### Device Mapping Strategy

```python
# Automatic layer distribution
device_map = infer_auto_device_map(
    model,
    max_memory={0: "70GB", 1: "70GB", "cpu": "200GB"},
    no_split_module_classes=["LlamaDecoderLayer"],  # Model-specific
    dtype=torch.float16
)

# Example result:
# GPU 0: Layers 0-40 (~70GB)
# GPU 1: Layers 41-80 (~70GB)
# CPU: Remaining layers (~50GB)
```

### Model-Specific Layer Classes

The system automatically detects layer types for:

- **Llama**: `LlamaDecoderLayer`
- **Falcon**: `FalconDecoderLayer`
- **BLOOM**: `BloomBlock`
- **GPT/StarCoder**: `GPTBigCodeBlock`, `GPTNeoXLayer`
- **Mistral**: `MistralDecoderLayer`

---

## Running Validation

### Quick Validation (10-25 minutes)

**Always run this BEFORE the full benchmark** to catch issues early:

```bash
# Validate your setup can load and run the model
python scripts/validate_large_model_setup.py \
    --config configs/production_config.yaml
```

**What it does:**
1. ✓ Checks GPU, RAM, and disk resources
2. ✓ Loads the large model from config
3. ✓ Tests a simple generation
4. ✓ Reports memory usage and time estimates

**Example output:**
```
✅ VALIDATION SUCCESSFUL!
Model Loading Time: 18.3 minutes
Generation Time: 87.2 seconds
Total Test Time: 19.8 minutes

📊 Benchmark Estimates:
  - 50 examples: ~91 minutes
  - 164 examples: ~256 minutes
```

---

## Running Benchmarks

### Option 1: Dedicated Large Model Benchmark

**Recommended for testing large models only:**

```bash
# Test with 50 examples, skipping code execution (faster)
python scripts/large_model_benchmark.py \
    --model llama3_405b \
    --num-examples 50 \
    --skip-execution \
    --config configs/production_config.yaml

# Full test with all 164 examples and code validation
python scripts/large_model_benchmark.py \
    --model llama3_405b \
    --num-examples 164 \
    --config configs/production_config.yaml

# Test smaller model for comparison
python scripts/large_model_benchmark.py \
    --model codellama_70b \
    --num-examples 50 \
    --skip-execution
```

**Features:**
- ✓ Pre-flight checks before loading model
- ✓ Progressive result saving (every 10 examples)
- ✓ Time estimation
- ✓ Memory usage monitoring
- ✓ Handles interruptions gracefully

### Option 2: Full System Comparison

**Compare RouterLLM (small models) vs Large Model (405B):**

```bash
# Enhanced comparison with 405B model
python scripts/enhanced_humaneval_comparison.py \
    --config configs/production_config.yaml \
    --num-examples 50 \
    --use-large-model \
    --skip-execution

# Full dataset comparison (164 examples)
python scripts/enhanced_humaneval_comparison.py \
    --config configs/production_config.yaml \
    --num-examples 164 \
    --use-large-model \
    --skip-execution
```

**This compares:**
- RouterLLM system (models 7B-70B with intelligent routing)
- Direct 405B model (single large model for all tasks)

**Metrics compared:**
- Accuracy / Generation success rate
- Latency (average time per example)
- Carbon footprint (CO2 emissions)
- Cost estimation

---

## Troubleshooting

### Out of Memory (OOM) Errors

#### Symptoms
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

#### Solutions (in order)

1. **Enable 4-bit quantization** (if not already):
   ```yaml
   use_4bit: true  # MANDATORY for 405B with 2-3 GPUs
   ```

2. **Increase CPU offload**:
   ```yaml
   max_memory:
     0: "65GB"      # Reduce GPU allocation
     1: "65GB"
     cpu: "250GB"   # Increase CPU offload
   ```

3. **Close other GPU processes**:
   ```bash
   # Check GPU usage
   nvidia-smi

   # Kill process using GPU
   kill -9 <PID>
   ```

4. **Use smaller model**:
   - Try Falcon-180B or BLOOM-176B instead of Llama-405B
   - Or use CodeLlama-70B for testing

5. **Disable Flash Attention**:
   ```yaml
   use_flash_attention: false  # Increases memory but may help loading
   ```

### Model Loading Timeout

#### Symptoms
- Loading takes longer than 30 minutes
- Process appears frozen

#### Solutions

1. **Check internet connection** - Initial download is ~800GB for 405B
2. **Use HuggingFace cache** - Set `HF_HOME` environment variable
3. **Monitor with logs** - Check `./logs/` directory for progress

### Slow Generation

#### Symptoms
- Generation takes >2 minutes per example
- Significantly slower than expected

#### Causes & Solutions

1. **CPU offload active** - Layers on CPU are slower
   - Solution: Use more GPUs or smaller model

2. **Disk swapping** - System running out of RAM
   - Solution: Close other applications, increase CPU memory allocation

3. **No Flash Attention** - Without FA2, attention is slower
   - Solution: Install flash-attn (requires CUDA 11.8+)

### HuggingFace Authentication Errors

#### Symptoms
```
Error: Model requires authentication
```

#### Solution

```bash
# Install huggingface-cli
pip install huggingface-hub

# Login with your token
huggingface-cli login

# Enter your token from https://huggingface.co/settings/tokens
```

Some models (Llama-3.1-405B) require:
1. Accepting the license on HuggingFace model page
2. Authenticated token with read access

---

## Performance Optimization

### Speed vs Memory Trade-offs

| Configuration | Loading Time | Inference Speed | VRAM Usage | CPU Usage |
|--------------|--------------|-----------------|------------|-----------|
| **FP16, 4 GPUs** | Fast (~10 min) | Fast (~30s) | Very High (800GB) | Low |
| **4-bit, 3 GPUs** | Medium (~18 min) | Medium (~60s) | High (200GB) | Medium |
| **4-bit, 2 GPUs + CPU** | Slow (~25 min) | Slow (~90s) | Medium (140GB) | High |
| **8-bit, 3 GPUs** | Medium (~15 min) | Medium (~45s) | High (400GB) | Low |

### Recommended Settings

#### For 2-3 GPU Setup (A100 80GB)
```yaml
use_4bit: true
use_flash_attention: true
max_memory: {0: "70GB", 1: "70GB", 2: "70GB", cpu: "200GB"}
offload_folder: "./offload"
```

#### For 4+ GPU Setup (A100 80GB)
```yaml
use_4bit: true
use_flash_attention: true
max_memory: {0: "75GB", 1: "75GB", 2: "75GB", 3: "75GB"}
# No CPU offload needed
```

### Monitoring Performance

```bash
# Monitor GPU usage (real-time)
watch -n 1 nvidia-smi

# Monitor CPU/RAM usage
htop

# Check disk I/O (if using offload)
iostat -x 1
```

---

## Best Practices

### Before Running Benchmark

1. **✓ Run validation script first**
   ```bash
   python scripts/validate_large_model_setup.py
   ```

2. **✓ Check available resources**
   ```bash
   nvidia-smi  # GPU memory
   free -h     # CPU RAM
   df -h       # Disk space
   ```

3. **✓ Clear GPU memory**
   ```bash
   # Restart system or kill GPU processes
   sudo fuser -v /dev/nvidia*
   ```

4. **✓ Set up HuggingFace cache**
   ```bash
   export HF_HOME="/path/to/large/disk/huggingface"
   ```

### During Benchmark

1. **Monitor resources** - Keep nvidia-smi running in another terminal
2. **Don't interrupt during loading** - First 15-25 min are critical
3. **Save progressively** - Results save every 10 examples automatically
4. **Use tmux/screen** - Run in background session to survive disconnects

### After Benchmark

1. **Clean up cache** - Remove temporary offload files
   ```bash
   rm -rf ./offload/*
   ```

2. **Clear GPU memory** - If running multiple benchmarks
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. **Analyze results** - Check `results/` directory for JSON outputs

### Cost Optimization

1. **Use skip-execution mode** - 20% faster, still useful for generation quality
2. **Test with subset first** - Start with 10-50 examples before full 164
3. **Use smaller model for testing** - CodeLlama-70B loads in 5 min
4. **Batch experiments** - Load model once, test multiple prompts

---

## Common Workflows

### Workflow 1: First-Time Setup

```bash
# 1. Validate setup
python scripts/validate_large_model_setup.py

# 2. Quick test with small model
python scripts/large_model_benchmark.py \
    --model codellama_70b \
    --num-examples 10 \
    --skip-execution

# 3. Full benchmark with large model
python scripts/large_model_benchmark.py \
    --model llama3_405b \
    --num-examples 50 \
    --skip-execution
```

### Workflow 2: Production Comparison

```bash
# 1. Validate setup
python scripts/validate_large_model_setup.py

# 2. Run full comparison (RouterLLM vs 405B)
python scripts/enhanced_humaneval_comparison.py \
    --config configs/production_config.yaml \
    --num-examples 164 \
    --use-large-model \
    --skip-execution \
    --results-dir results/final_comparison
```

### Workflow 3: Debugging Issues

```bash
# 1. Check system resources
nvidia-smi
free -h
df -h

# 2. Run validation with detailed logs
python scripts/validate_large_model_setup.py \
    --config configs/production_config.yaml \
    2>&1 | tee validation.log

# 3. If OOM, try smaller model or adjust config
# Edit production_config.yaml:
#   - Reduce max_memory per GPU
#   - Increase CPU offload
#   - Enable 4-bit if not already
```

---

## Additional Resources

### Official Documentation

- **Accelerate**: https://huggingface.co/docs/accelerate
- **Transformers**: https://huggingface.co/docs/transformers
- **BitsAndBytes**: https://github.com/TimDettmers/bitsandbytes

### Useful Links

- **HuggingFace Models**: https://huggingface.co/models
- **Llama-3.1-405B**: https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct
- **Model Memory Calculator**: https://huggingface.co/spaces/hf-accelerate/model-memory-usage

### Project Files

- **Main config**: `configs/production_config.yaml`
- **Large model manager**: `src/routerllm/models/large_model_manager.py`
- **Validation script**: `scripts/validate_large_model_setup.py`
- **Benchmark script**: `scripts/large_model_benchmark.py`
- **Comparison script**: `scripts/enhanced_humaneval_comparison.py`

---

## FAQ

### Q: Can I run 405B model on 2 GPUs?

**A:** Yes, but with significant CPU offload. Expect:
- Loading: 25-30 minutes
- Generation: 90-120 seconds per example
- Requires 200GB+ CPU RAM

### Q: What's the difference between skip-execution and full validation?

**A:**
- **Skip-execution**: Only generates code, doesn't run it. Faster (~20%), measures generation quality
- **Full validation**: Runs generated code with test cases. Slower, measures actual correctness

### Q: How much does it cost to run the benchmark?

**A:**
- **On-premise**: Electricity + GPU depreciation (~$50-100 per benchmark)
- **Cloud (AWS p4d.24xlarge)**: ~$32/hour × 4-5 hours = $128-160
- **Colab/Cloud with spot instances**: Can be 60-80% cheaper

### Q: Can I use multiple large models in parallel?

**A:** Not recommended. Each 405B model needs 2-3 GPUs. Run benchmarks sequentially.

### Q: What if validation succeeds but benchmark fails?

**A:** Likely causes:
1. Dataset download issues (check internet)
2. Disk space filled during benchmark
3. OOM due to long-running process (restart before benchmark)

---

## Support

For issues:
1. Check logs in `./logs/` directory
2. Review error messages carefully
3. Try validation script first
4. Consult troubleshooting section above
5. Open GitHub issue with logs and config

---

**Last Updated**: 2025-11-07
**System Version**: RouterLLM v2.0 with Accelerate Integration
