# RouterLLM

Intelligent LLM Router for optimizing accuracy/cost efficiency with carbon footprint tracking.

## Overview

RouterLLM is a system that intelligently routes text prompts to the most appropriate Large Language Model (LLM) based on the nature of the request. The system aims to optimize the balance between response quality, computational cost, and energy consumption.

## Features

- **Intelligent Routing**: BERT-based router that classifies requests and selects optimal LLM
- **Multiple LLM Support**: Manages multiple Hugging Face models with automatic loading/unloading
- **Carbon Footprint Tracking**: Comprehensive emission monitoring using CodeCarbon
- **Flexible Architecture**: Easy switching between dummy and trained routers
- **Memory Optimization**: GPU memory management with quantization support
- **Comprehensive Logging**: Detailed performance and carbon emission logs

## Quick Start

### 1. Generate Training Data

```bash
python main.py generate-data --samples 1200 --output-dir ./data
```

### 2. Train the Router (Optional)

```bash
python main.py train --data-dir ./data --model-dir ./models --epochs 3
```

### 3. Test the System

```bash
# Test with dummy router
python main.py demo --router-type dummy

# Test with trained BERT router
python main.py demo --router-type bert --router-model ./models/best_router.pt
```

## Usage Example

```python
from src.routerllm.core.system import RouterLLMSystem

# Initialize system
system = RouterLLMSystem(
    config_path="configs/default_config.yaml",
    router_type="dummy",  # or "bert"
    enable_carbon_tracking=True
)

system.initialize()

# Generate response
result = system.predict_and_generate(
    "Write a Python function to calculate factorial"
)

print(f"Selected Model: {result['predicted_model']}")
print(f"Response: {result['response']}")

system.cleanup()
```

## Architecture

The system consists of:

1. **Router Models**: BERT-based or dummy router for LLM selection
2. **LLM Manager**: Handles multiple Hugging Face models
3. **Training System**: Standard and Inter-Intra loss training
4. **Carbon Tracking**: Real-time emission monitoring
5. **Data Generation**: Synthetic dataset generation

## LLM Categories

- **Category 0 (Code Generation)**: Programming, debugging, algorithms
- **Category 1 (Text Generation)**: Creative writing, content creation
- **Category 2 (General Purpose)**: Analysis, explanations, advice
- **Category 3 (Lightweight Tasks)**: Simple questions, translations

## Supported Models (<15B parameters)

- **CodeLlama-7B**: Code generation
- **Phi-3-Mini**: Text generation
- **Mistral-7B**: General purpose
- **TinyLlama-1.1B**: Lightweight tasks