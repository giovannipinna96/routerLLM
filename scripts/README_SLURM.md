# RouterLLM SLURM Pipeline Scripts

This directory contains scripts for running the complete RouterLLM pipeline on a SLURM cluster.

## 📁 Files

- **`run_full_pipeline.slurm`** - Main SLURM script that runs the entire pipeline
- **`test_router_and_save.py`** - Python script to test a router and save results to JSON
- **`aggregate_results.py`** - Python script to aggregate JSON results into CSV

---

## 🚀 Quick Start

### Default Run (50k samples)

```bash
sbatch scripts/run_full_pipeline.slurm
```

This will:
- Generate 10,000 synthetic samples
- Load 50,000 samples from tiny-codes dataset
- Train BERT router on both datasets
- Test 4 routers (BERT x2, Dummy, Graham)
- Save results to `./results/`

### Full Dataset Run (1.6M samples)

```bash
sbatch --export=DATASET_SIZE=full scripts/run_full_pipeline.slurm
```

This will use the complete tiny-codes dataset (1.6M samples). **Note: This will take 20-24 hours!**

---

## 📊 Pipeline Steps

The SLURM script executes the following pipeline:

### STEP 1: Generate Synthetic Dataset
- Creates synthetic training data for router
- Output: `./data/synthetic/train_dataset.json` (+ val + test)

### STEP 2: Load HuggingFace Dataset
- Downloads and processes tiny-codes dataset
- Output: `./data/tinycodes/train_dataset.json` (+ val + test)

### STEP 3: Train BERT Router (Synthetic)
- Trains BERT router on synthetic data
- 3 epochs, batch size 16
- Output: `./models/bert_synthetic/best_router.pt`

### STEP 4: Train BERT Router (Tiny-Codes)
- Trains BERT router on tiny-codes data
- 3 epochs, batch size 16
- Output: `./models/bert_tinycodes/best_router.pt`

### STEP 5: Test All Routers
Tests 4 routers with 15 predefined prompts:
1. **BERT (Synthetic)** - BERT trained on synthetic data
2. **BERT (Tiny-Codes)** - BERT trained on HuggingFace data
3. **Dummy** - Random selection baseline
4. **Graham Complexity** - Pre-trained complexity classifier

Each test saves:
- Performance metrics (accuracy, latency)
- Model selection statistics
- Carbon footprint breakdown
- Output: `./results/<router_name>_results.json`

### STEP 6: Aggregate Results
- Combines all JSON results into single CSV
- Output: `./results/router_comparison.csv`

---

## 📂 Output Structure

After running the pipeline, you'll have:

```
./data/
├── synthetic/
│   ├── train_dataset.json
│   ├── val_dataset.json
│   └── test_dataset.json
└── tinycodes/
    ├── train_dataset.json
    ├── val_dataset.json
    └── test_dataset.json

./models/
├── bert_synthetic/
│   └── best_router_epoch_X.pt
└── bert_tinycodes/
    └── best_router_epoch_X.pt

./results/
├── bert_synthetic_results.json
├── bert_tinycodes_results.json
├── dummy_results.json
├── graham_results.json
└── router_comparison.csv

./logs/
├── carbon/
│   └── emissions.csv
└── slurm_<jobid>.out
```

---

## ⚙️ Configuration Options

### SLURM Parameters

Edit `run_full_pipeline.slurm` to customize:

```bash
#SBATCH --time=24:00:00        # Max runtime (24 hours)
#SBATCH --cpus-per-task=8      # CPU cores
#SBATCH --mem=64G              # RAM
#SBATCH --gpus=1               # Number of GPUs
#SBATCH --partition=gpu        # SLURM partition
```

### Dataset Size

Control dataset size with environment variable:

```bash
# 50k samples (default)
sbatch scripts/run_full_pipeline.slurm

# Full 1.6M samples
sbatch --export=DATASET_SIZE=full scripts/run_full_pipeline.slurm
```

**Dataset size mapping:**

| `DATASET_SIZE` | Synthetic Samples | Tiny-Codes Samples | Est. Time |
|----------------|-------------------|-------------------|-----------|
| `50k` (default) | 10,000 | 50,000 | 2-4 hours |
| `full` | 50,000 | 1,600,000 | 20-24 hours |

### Training Parameters

Edit in `run_full_pipeline.slurm`:

```bash
EPOCHS=3           # Number of training epochs
BATCH_SIZE=16      # Batch size for training
```

---

## 🔬 Using Individual Scripts

### Test a Single Router

```bash
# Test BERT router
uv run python scripts/test_router_and_save.py \
    --router-type bert \
    --router-model ./models/bert_synthetic/best_router.pt \
    --output-file ./results/my_test.json \
    --router-name "My_BERT_Test"

# Test Dummy router (no model needed)
uv run python scripts/test_router_and_save.py \
    --router-type dummy \
    --output-file ./results/dummy_test.json
```

### Aggregate Custom Results

```bash
uv run python scripts/aggregate_results.py \
    --results-dir ./results \
    --output-csv ./results/custom_comparison.csv \
    --verbose
```

---

## 📈 Analyzing Results

### JSON Results

Each router generates a detailed JSON file with:

```json
{
  "router_name": "BERT_Synthetic",
  "router_type": "bert",
  "summary_stats": {
    "total_prompts": 15,
    "successful": 15,
    "success_rate": 1.0,
    "average_total_time": 2.4567,
    "model_selections": {
      "codellama_13b": 5,
      "mistral_7b": 3,
      "phi3_mini": 7
    }
  },
  "carbon_footprint": {
    "total_emissions_kg": 0.000234,
    "emissions_breakdown": {
      "router_inference": 0.000012,
      "model_loading": 0.000089,
      "llm_inference": 0.000133
    }
  }
}
```

### CSV Comparison

The aggregated CSV contains:

| Column | Description |
|--------|-------------|
| Router Name | Human-readable router name |
| Success Rate (%) | Percentage of successful requests |
| Avg Total Time (s) | Average end-to-end latency |
| Most Selected Model | LLM selected most often |
| Total CO2 Emissions (kg) | Total carbon footprint |
| Avg CO2 per Request (kg) | Average emissions per request |

**Open in Excel/LibreOffice:**
```bash
libreoffice results/router_comparison.csv
```

**Analyze with pandas:**
```python
import pandas as pd

df = pd.read_csv('results/router_comparison.csv')
print(df.sort_values('Total CO2 Emissions (kg)'))
```

---

## 🐛 Troubleshooting

### Job Failed to Start

Check SLURM logs:
```bash
tail -f logs/slurm_<jobid>.err
```

Common issues:
- **GPU not available**: Check `#SBATCH --gpus=1` matches your cluster
- **Partition not found**: Update `#SBATCH --partition=gpu` to your cluster's partition name
- **Time limit**: Increase `#SBATCH --time=` for full dataset

### Out of Memory

Reduce batch size in `run_full_pipeline.slurm`:
```bash
BATCH_SIZE=8  # Instead of 16
```

Or request more memory:
```bash
#SBATCH --mem=128G  # Instead of 64G
```

### HuggingFace Dataset Download Fails

Login to HuggingFace first:
```bash
huggingface-cli login
```

Or set token in environment:
```bash
export HF_TOKEN="your_token_here"
sbatch scripts/run_full_pipeline.slurm
```

### Model Loading Errors

Check available GPU memory:
```bash
nvidia-smi
```

Models use 4-bit quantization by default. If still OOM, edit `configs/default_config.yaml` to use smaller models.

---

## 📊 Monitoring Running Jobs

### Check Job Status
```bash
squeue -u $USER
```

### View Live Output
```bash
tail -f logs/slurm_<jobid>.out
```

### Cancel Job
```bash
scancel <jobid>
```

### Check GPU Usage
```bash
srun --jobid=<jobid> nvidia-smi
```

---

## 🔧 Customization

### Add Custom Test Prompts

Edit `test_router_and_save.py` and modify `TEST_PROMPTS` list:

```python
TEST_PROMPTS = [
    "Your custom prompt 1",
    "Your custom prompt 2",
    # ...
]
```

### Test Different Routers

To add MoE or RL routers, modify STEP 5 in `run_full_pipeline.slurm`:

```bash
# Test MoE router (if trained)
uv run python "${SCRIPTS_DIR}/test_router_and_save.py" \
    --router-type moe \
    --router-model ./models/moe_router.pt \
    --output-file "${RESULTS_DIR}/moe_results.json"
```

### Change LLM Models

Edit `configs/default_config.yaml` to customize which LLMs are available for routing.

---

## 📝 Notes

- **GPU Required**: All steps after data generation require GPU
- **Disk Space**: Full dataset requires ~100GB disk space
- **HuggingFace Cache**: Models are cached in `~/.cache/huggingface/` (~50-200GB)
- **Carbon Tracking**: Automatically enabled, logs to `./logs/carbon/emissions.csv`
- **No Large Models**: This pipeline uses standard models (up to 70B). For 100B+ models, see separate documentation.

---

## 📧 Support

For issues or questions:
1. Check SLURM logs: `logs/slurm_<jobid>.err`
2. Check RouterLLM logs: `logs/routerllm.log`
3. Review test results: `results/*_results.json`

---

## 🎯 Example Workflow

```bash
# 1. Submit job with 50k samples for quick test
sbatch scripts/run_full_pipeline.slurm

# 2. Monitor progress
tail -f logs/slurm_*.out

# 3. Check results when complete
cat results/router_comparison.csv

# 4. If satisfied, run full dataset
sbatch --export=DATASET_SIZE=full scripts/run_full_pipeline.slurm
```

---

**Happy Routing!** 🚀
