# RouterLLM Usage Guide

Guida pratica per runnare il sistema RouterLLM in tutte le modalità disponibili.

---

## 📋 Indice

- [Quick Start](#quick-start)
- [Validazione Setup](#validazione-setup)
- [Benchmark Large Model Solo](#benchmark-large-model-solo)
- [Confronto Router vs Large Model](#confronto-router-vs-large-model)
- [Training Router](#training-router)
- [Testing Modelli Singoli](#testing-modelli-singoli)
- [Modalità Avanzate](#modalità-avanzate)

---

## 🚀 Quick Start

### 1. Validazione Setup (SEMPRE PRIMA!)

```bash
# Verifica che il tuo hardware possa caricare modelli large (150B+)
# Tempo: 20-30 minuti
python scripts/validate_large_model_setup.py --config configs/production_config.yaml
```

**Output atteso:**
```
✅ VALIDATION SUCCESSFUL!
Model Loading Time: 18.3 minutes
Generation Time: 87.2 seconds
📊 Benchmark Estimates:
  - 50 examples: ~91 minutes
  - 164 examples: ~256 minutes
```

---

## 🔍 Validazione Setup

### Validazione Completa

```bash
# Con config production (Llama-3.1-405B)
python scripts/validate_large_model_setup.py \
    --config configs/production_config.yaml
```

### Validazione con Config Custom

```bash
# Con config diverso
python scripts/validate_large_model_setup.py \
    --config configs/my_custom_config.yaml
```

**Cosa fa:**
- ✓ Controlla GPU, RAM, Disk disponibili
- ✓ Carica il large model configurato
- ✓ Testa una generazione semplice
- ✓ Stima tempi per 50 e 164 esempi
- ✓ Mostra uso memoria GPU/CPU

---

## 🎯 Benchmark Large Model Solo

Testa SOLO il large model (senza confronto con router).

### Test Rapido - 10 Esempi

```bash
# Test velocissimo per verificare che funzioni
# Tempo: ~30-40 minuti
python scripts/large_model_benchmark.py \
    --model llama3_405b \
    --num-examples 10 \
    --skip-execution
```

### Test Medio - 50 Esempi

```bash
# Test bilanciato
# Tempo: ~1.5-2 ore
python scripts/large_model_benchmark.py \
    --model llama3_405b \
    --num-examples 50 \
    --skip-execution \
    --results-dir results/test_50
```

### Test Completo - 164 Esempi (Tutti HumanEval)

```bash
# Benchmark completo con TUTTI gli esempi
# Tempo: ~4-5 ore
python scripts/large_model_benchmark.py \
    --model llama3_405b \
    --num-examples 164 \
    --skip-execution \
    --results-dir results/full_164
```

### Test con Validazione Codice (Più Lento)

```bash
# SENZA --skip-execution esegue il codice generato
# Tempo: ~5-7 ore per 164 esempi
python scripts/large_model_benchmark.py \
    --model llama3_405b \
    --num-examples 164 \
    --results-dir results/validated
```

### Test con Modelli Alternativi

```bash
# Falcon-180B (più veloce)
python scripts/large_model_benchmark.py \
    --model falcon_180b \
    --num-examples 50 \
    --skip-execution

# BLOOM-176B
python scripts/large_model_benchmark.py \
    --model bloom_176b \
    --num-examples 50 \
    --skip-execution

# CodeLlama-70B (molto più veloce, per testing)
python scripts/large_model_benchmark.py \
    --model codellama_70b \
    --num-examples 50 \
    --skip-execution
```

### Saltare Pre-flight Checks (Non Consigliato)

```bash
# Solo se sei SICURO che tutto funziona
python scripts/large_model_benchmark.py \
    --model llama3_405b \
    --num-examples 50 \
    --skip-execution \
    --skip-preflight
```

### Con Seed Custom

```bash
# Per riproducibilità con seed diverso
python scripts/large_model_benchmark.py \
    --model llama3_405b \
    --num-examples 50 \
    --skip-execution \
    --seed 123
```

---

## ⚖️ Confronto Router vs Large Model

Confronta il sistema RouterLLM (router intelligente + modelli <70B) contro Large Model 405B diretto.

### Confronto Rapido - 50 Esempi

```bash
# Confronto completo con 50 esempi
# Tempo: ~2-3 ore
python scripts/enhanced_humaneval_comparison.py \
    --config configs/production_config.yaml \
    --num-examples 50 \
    --use-large-model \
    --skip-execution \
    --results-dir results/comparison_50
```

### Confronto Completo - 164 Esempi

```bash
# Benchmark definitivo con tutti gli esempi
# Tempo: ~6-8 ore
python scripts/enhanced_humaneval_comparison.py \
    --config configs/production_config.yaml \
    --num-examples 164 \
    --use-large-model \
    --skip-execution \
    --results-dir results/comparison_164
```

### Confronto con Router MoE Dinamico

```bash
# Usa router MoE invece di complexity router
python scripts/enhanced_humaneval_comparison.py \
    --config configs/production_config.yaml \
    --num-examples 50 \
    --use-large-model \
    --use-dynamic-router \
    --skip-execution
```

### Confronto con Validazione Codice

```bash
# SENZA --skip-execution esegue e valida il codice
# Tempo: ~8-12 ore per 164 esempi
python scripts/enhanced_humaneval_comparison.py \
    --config configs/production_config.yaml \
    --num-examples 164 \
    --use-large-model \
    --results-dir results/validated_comparison
```

### Confronto Solo Router (Senza Large Model)

```bash
# Testa solo il router system
python scripts/enhanced_humaneval_comparison.py \
    --config configs/default_config.yaml \
    --num-examples 50 \
    --skip-execution
```

### Con Config Diverso

```bash
# Usa config custom
python scripts/enhanced_humaneval_comparison.py \
    --config configs/my_config.yaml \
    --num-examples 50 \
    --use-large-model \
    --skip-execution
```

---

## 🎓 Training Router

### 1. Genera Dataset Sintetico

```bash
# Genera 1200 esempi per training
# Tempo: ~5-10 minuti
python main.py generate-data \
    --samples 1200 \
    --output-dir ./data
```

**Output:**
```
data/
├── train.json      # 840 esempi (70%)
├── val.json        # 180 esempi (15%)
└── test.json       # 180 esempi (15%)
```

### 2. Train BERT Router

```bash
# Training base
# Tempo: ~30-60 minuti
python main.py train \
    --data-dir ./data \
    --model-dir ./models \
    --epochs 3 \
    --batch-size 16
```

### 3. Train con Inter-Intra Loss

```bash
# Training avanzato con loss customizzata
python main.py train \
    --data-dir ./data \
    --model-dir ./models \
    --epochs 5 \
    --batch-size 16 \
    --use-inter-intra-loss \
    --learning-rate 2e-5
```

### 4. Train con Più Epochs

```bash
# Training lungo per miglior accuratezza
python main.py train \
    --data-dir ./data \
    --model-dir ./models \
    --epochs 10 \
    --batch-size 32 \
    --use-inter-intra-loss
```

---

## 🧪 Testing Modelli Singoli

### Test Dummy Router (No Training Needed)

```bash
# Router casuale per testing rapido
python main.py demo --router-type dummy
```

### Test BERT Router (Dopo Training)

```bash
# Usa router BERT trainato
python main.py test \
    --router-type bert \
    --router-model ./models/best_router.pt \
    --test-examples
```

### Test Graham Complexity Router

```bash
# Router pre-trainato basato su complessità
python main.py demo --router-type graham_complexity
```

### Test MoE Router

```bash
# Mixture-of-Experts router
python main.py demo --router-type moe
```

### Test RL Router

```bash
# Reinforcement Learning router
python main.py demo --router-type rl --carbon-tracking
```

### Test Integrated System

```bash
# Sistema integrato con tutte le features
python main.py demo \
    --router-type integrated \
    --carbon-tracking \
    --carbon-optimization balanced \
    --enable-large-models
```

---

## 🔧 Modalità Avanzate

### Con Carbon Tracking

```bash
# Traccia emissioni CO2
python scripts/large_model_benchmark.py \
    --model llama3_405b \
    --num-examples 50 \
    --skip-execution

# I risultati includeranno:
# - Total CO2 emissions (kg)
# - CO2 per example
# - Breakdown per componente
```

### Background Execution con tmux/screen

```bash
# Per sessioni lunghe che sopravvivono a disconnessioni

# Con tmux
tmux new -s benchmark
python scripts/large_model_benchmark.py \
    --model llama3_405b \
    --num-examples 164 \
    --skip-execution
# Ctrl+B poi D per detach

# Riattacca con
tmux attach -t benchmark

# Con screen
screen -S benchmark
python scripts/large_model_benchmark.py \
    --model llama3_405b \
    --num-examples 164 \
    --skip-execution
# Ctrl+A poi D per detach

# Riattacca con
screen -r benchmark
```

### Monitoring Risorse Durante Execution

```bash
# Terminale 1: Run benchmark
python scripts/large_model_benchmark.py \
    --model llama3_405b \
    --num-examples 50 \
    --skip-execution

# Terminale 2: Monitor GPU
watch -n 1 nvidia-smi

# Terminale 3: Monitor CPU/RAM
htop

# Terminale 4: Monitor Disk I/O
iostat -x 1
```

### Logging Dettagliato

```bash
# Salva logs in file
python scripts/large_model_benchmark.py \
    --model llama3_405b \
    --num-examples 50 \
    --skip-execution \
    2>&1 | tee benchmark_$(date +%Y%m%d_%H%M%S).log
```

### Con Variabili Ambiente

```bash
# Set HuggingFace cache directory
export HF_HOME="/mnt/large_disk/huggingface"

# Set CUDA devices (usa solo GPU 0 e 1)
export CUDA_VISIBLE_DEVICES=0,1

# Run benchmark
python scripts/large_model_benchmark.py \
    --model llama3_405b \
    --num-examples 50 \
    --skip-execution
```

### Pulizia Cache tra Runs

```bash
# Prima del benchmark
rm -rf ./offload/*
rm -rf ./logs/carbon/*

# Pulizia cache HuggingFace (attenzione: ri-scarica modelli!)
# rm -rf ~/.cache/huggingface/

# Run benchmark
python scripts/large_model_benchmark.py \
    --model llama3_405b \
    --num-examples 50 \
    --skip-execution
```

---

## 📊 Analisi Risultati

### Struttura Risultati

```
results/
├── large_model_benchmark/
│   ├── llama3_405b_partial_10.json    # Salvataggio progressivo
│   ├── llama3_405b_partial_20.json
│   ├── llama3_405b_partial_50.json
│   └── llama3_405b_final_50.json      # Risultati finali
│
├── comparison_50/
│   ├── router_results.json             # Risultati RouterLLM
│   ├── direct_results.json             # Risultati Large Model
│   └── comparison_analysis.json        # Analisi comparativa
│
└── logs/
    └── carbon/
        └── emissions.csv               # Tracking CO2
```

### Visualizza Risultati Finali

```bash
# Visualizza JSON formattato
cat results/large_model_benchmark/llama3_405b_final_50.json | jq '.performance'

# Estrai accuratezza
cat results/large_model_benchmark/llama3_405b_final_50.json | jq '.performance.accuracy'

# Vedi emissioni CO2
cat results/large_model_benchmark/llama3_405b_final_50.json | jq '.system_stats.carbon_footprint'
```

### Confronto Risultati

```bash
# Vedi confronto finale
cat results/comparison_50/comparison_analysis.json | jq '.conclusion'

# Accuratezza router vs large model
cat results/comparison_50/comparison_analysis.json | jq '.accuracy_comparison'

# Carbon footprint comparison
cat results/comparison_50/comparison_analysis.json | jq '.carbon_footprint_comparison'
```

---

## 🎯 Workflow Completi

### Workflow 1: Prima Volta - Setup e Test

```bash
# Step 1: Valida setup (20-30 min)
python scripts/validate_large_model_setup.py

# Step 2: Test rapido con 10 esempi (30-40 min)
python scripts/large_model_benchmark.py \
    --model llama3_405b \
    --num-examples 10 \
    --skip-execution

# Step 3: Se OK, test medio con 50 esempi (1.5-2 ore)
python scripts/large_model_benchmark.py \
    --model llama3_405b \
    --num-examples 50 \
    --skip-execution \
    --results-dir results/test_50
```

### Workflow 2: Benchmark Completo

```bash
# Step 1: Valida setup
python scripts/validate_large_model_setup.py

# Step 2: Run full benchmark 164 esempi (4-5 ore)
# In tmux per sicurezza
tmux new -s benchmark
python scripts/large_model_benchmark.py \
    --model llama3_405b \
    --num-examples 164 \
    --skip-execution \
    --results-dir results/full_benchmark \
    2>&1 | tee full_benchmark.log
```

### Workflow 3: Confronto Completo Router vs Large Model

```bash
# Step 1: Valida setup
python scripts/validate_large_model_setup.py

# Step 2: (Opzionale) Train router se necessario
python main.py generate-data --samples 1200 --output-dir ./data
python main.py train --data-dir ./data --model-dir ./models --epochs 5

# Step 3: Run confronto completo (6-8 ore)
tmux new -s comparison
python scripts/enhanced_humaneval_comparison.py \
    --config configs/production_config.yaml \
    --num-examples 164 \
    --use-large-model \
    --skip-execution \
    --results-dir results/final_comparison \
    2>&1 | tee comparison.log
```

### Workflow 4: Test Multipli Modelli

```bash
# Testa tutti i modelli disponibili per confronto

# Modello 1: Llama-3.1-405B
python scripts/large_model_benchmark.py \
    --model llama3_405b \
    --num-examples 50 \
    --skip-execution \
    --results-dir results/llama_405b

# Modello 2: Falcon-180B
python scripts/large_model_benchmark.py \
    --model falcon_180b \
    --num-examples 50 \
    --skip-execution \
    --results-dir results/falcon_180b

# Modello 3: CodeLlama-70B
python scripts/large_model_benchmark.py \
    --model codellama_70b \
    --num-examples 50 \
    --skip-execution \
    --results-dir results/codellama_70b

# Confronta risultati
ls -lh results/*/
```

---

## ⚠️ Troubleshooting Comuni

### Out of Memory

```bash
# Se ricevi OOM error, usa modello più piccolo per test
python scripts/large_model_benchmark.py \
    --model codellama_70b \
    --num-examples 10 \
    --skip-execution
```

### HuggingFace Authentication

```bash
# Se modello richiede autenticazione
huggingface-cli login

# Inserisci token da https://huggingface.co/settings/tokens
```

### Disk Space Full

```bash
# Pulisci offload directory
rm -rf ./offload/*

# Pulisci logs vecchi
rm -rf ./logs/*.log

# Controlla spazio disco
df -h
```

### GPU Occupata

```bash
# Vedi processi GPU
nvidia-smi

# Killa processo specifico
kill -9 <PID>

# Pulisci cache GPU
python -c "import torch; torch.cuda.empty_cache()"
```

---

## 📚 Documentazione Aggiuntiva

- **Setup Dettagliato**: `docs/LARGE_MODEL_SETUP.md`
- **Istruzioni Progetto**: `CLAUDE.md`
- **README**: `README.md`

---

## 🆘 Help

### Opzioni Available per Ogni Script

```bash
# Large model benchmark
python scripts/large_model_benchmark.py --help

# Enhanced comparison
python scripts/enhanced_humaneval_comparison.py --help

# Validation
python scripts/validate_large_model_setup.py --help

# Main script
python main.py --help
```

---

## 🎉 Comandi Più Usati (Cheat Sheet)

```bash
# 1. VALIDA SEMPRE PRIMA
python scripts/validate_large_model_setup.py

# 2. TEST RAPIDO (10 esempi)
python scripts/large_model_benchmark.py --model llama3_405b --num-examples 10 --skip-execution

# 3. TEST MEDIO (50 esempi)
python scripts/large_model_benchmark.py --model llama3_405b --num-examples 50 --skip-execution

# 4. TEST COMPLETO (164 esempi)
python scripts/large_model_benchmark.py --model llama3_405b --num-examples 164 --skip-execution

# 5. CONFRONTO ROUTER VS LARGE (50 esempi)
python scripts/enhanced_humaneval_comparison.py --num-examples 50 --use-large-model --skip-execution

# 6. CONFRONTO COMPLETO (164 esempi)
python scripts/enhanced_humaneval_comparison.py --num-examples 164 --use-large-model --skip-execution
```

---

**Last Updated**: 2025-11-07
**Version**: RouterLLM v2.0 con Accelerate Integration
