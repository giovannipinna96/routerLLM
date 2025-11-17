# 🚀 Guida Completa al Sistema Integrato RouterLLM

## Panoramica

Il sistema integrato RouterLLM ora supporta **7 strategie di routing avanzate** con ottimizzazione carbon-aware, supporto per modelli 100B+ e gestione intelligente delle risorse.

---

## 📊 Strategie di Routing Disponibili

### 1. **DUMMY** - Router Casuale (Baseline)
```bash
uv run python main.py demo --router-type dummy
```

**Uso**: Testing e sviluppo rapido

### 2. **BERT** - Classificatore Trained
```bash
# Train router first
uv run python main.py generate-data --samples 1200 --output-dir ./data
uv run python main.py train --data-dir ./data --model-dir ./models --epochs 3

# Use trained router
uv run python main.py demo --router-type bert --router-model ./models/best_router.pt
```

**Uso**: Routing basato su classificazione del prompt

### 3. **COMPLEXITY** - Routing Basato su Complessità
```bash
uv run python main.py demo --router-type graham_complexity
```

**Uso**: Seleziona modello in base alla complessità stimata del prompt

### 4. **MOE** - Dynamic Mixture of Experts ⭐ NUOVO
```bash
uv run python main.py demo --router-type moe
```

**Features**:
- Gating network neurale
- Sparse gating (top-k selection)
- Load balancing automatico
- Carbon & cost aware

**Uso**: Routing dinamico con selezione multipla di expert

### 5. **RL** - Reinforcement Learning Router ⭐ NUOVO
```bash
uv run python main.py demo --router-type rl --carbon-tracking
```

**Features**:
- Dueling DQN architecture
- Multi-objective optimization
- Carbon-aware exploration
- Online learning capability

**Uso**: Routing ottimizzato con apprendimento continuo

### 6. **INTEGRATED** - Sistema Completo Carbon-Aware ⭐ NUOVO
```bash
uv run python main.py demo --router-type integrated \
    --carbon-tracking \
    --carbon-optimization balanced \
    --enable-large-models
```

**Features**:
- Combina tutte le strategie
- Carbon budget management
- Supporto modelli 100B+
- Request caching e batching

**Uso**: Deployment produzione con tutte le ottimizzazioni

---

## 🎯 Come Usare i Nuovi Router

### Quick Start - MoE Router

```python
from src.routerllm.models.moe_router import DynamicMoERouter

# Initialize router
router = DynamicMoERouter(
    encoder_model="bert-base-uncased",
    num_experts=4,
    hidden_dim=256,
    top_k=2,
    carbon_aware=True,
    cost_aware=True
)

# Route a request
result = router.forward(
    text="Implement a distributed cache system",
    return_all_scores=True
)

print(f"Selected Expert: {result['expert_name']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Estimated Carbon: {result['estimated_carbon']:.6f} kg")
print(f"Estimated Cost: ${result['estimated_cost']:.6f}")
```

### Training MoE Router

```python
# Prepare training data: (text, optimal_expert, reward)
train_data = [
    ("Write complex algorithm", 0, 0.95),  # Heavy model
    ("Simple query", 3, 0.85),             # Light model
    ("General question", 2, 0.80),         # Medium model
    # ... more examples
]

# Train
router.train_gating_network(
    train_data=train_data,
    num_epochs=10,
    learning_rate=1e-4,
    batch_size=32
)

# Save trained router
router.save_model("models/moe_router.pt")
```

### Quick Start - RL Router

```python
from src.routerllm.models.rl_router import ReinforcementLearningRouter, RLConfig

# Configure RL router
config = RLConfig(
    accuracy_weight=0.4,
    carbon_weight=0.3,
    cost_weight=0.2,
    latency_weight=0.1,
    enable_carbon_aware_exploration=True
)

# Initialize
router = ReinforcementLearningRouter(
    config=config,
    encoder_model="microsoft/codebert-base"
)

# Make routing decision
result = router.forward(
    text="Create a machine learning model",
    training=False,
    carbon_budget=0.005  # kg CO2
)

print(f"Selected Expert: {result['expert_name']}")
print(f"Q-Values: {result['q_values']}")
print(f"Expected Carbon: {result['expected_carbon']:.6f} kg")

# Get carbon report
report = router.get_carbon_report()
print(f"Total Emissions: {report['total_emissions_kg']:.6f} kg")
print(f"Carbon Saved: {report['carbon_saved_kg']:.6f} kg")
print(f"Efficiency Score: {report['carbon_efficiency_score']:.4f}")
```

### Training RL Router

```python
from src.routerllm.models.rl_router import RLTrainer

# Prepare training data
train_data = [
    ("text", optimal_expert_id, base_accuracy),
    # ... more examples
]

# Create trainer
trainer = RLTrainer(
    router=router,
    train_data=train_data
)

# Train
trainer.train_episode(num_episodes=100)

# Save trained router
router.save_model("models/rl_router.pt")
```

### Quick Start - Sistema Integrato

```python
from src.routerllm.core.integrated_system import (
    IntegratedRouterLLMSystem,
    RouterStrategy,
    SystemConfig
)

# Configure system
config = SystemConfig(
    router_strategy=RouterStrategy.CARBON_AWARE,
    enable_carbon_optimization=True,
    carbon_optimization_level="balanced",  # aggressive/balanced/relaxed
    enable_100b_models=True,
    enable_caching=True,
    enable_batching=True
)

# Initialize
system = IntegratedRouterLLMSystem(
    config_path="configs/production_config.yaml",
    system_config=config
)

system.initialize()

# Process requests
result = system.process_request(
    text="Implement blockchain consensus",
    strategy_override=RouterStrategy.ENSEMBLE  # Can override strategy per request
)

print(f"Model Used: {result['model_used']}")
print(f"Response: {result['response']}")
print(f"Carbon: {result['carbon_emissions_kg']} kg")

# Get comprehensive report
report = system.get_system_report()
print(f"Total Requests: {report['statistics']['total_requests']}")
print(f"Cache Hit Rate: {report['statistics']['cache_hit_rate']:.2%}")
print(f"Carbon Saved: {report['statistics']['carbon_saved_kg']} kg")
```

---

## 🌱 Carbon Optimization

### Livelli di Ottimizzazione

#### Aggressive
- **Carbon Weight**: 50%
- **Quality Weight**: 30%
- **Max Model**: 34B
- **Use Case**: Massima riduzione emissioni

```bash
uv run python main.py demo --router-type integrated \
    --carbon-optimization aggressive
```

#### Balanced (Raccomandato)
- **Carbon Weight**: 35%
- **Quality Weight**: 45%
- **Max Model**: 70B
- **Use Case**: Bilanciamento qualità/emissioni

```bash
uv run python main.py demo --router-type integrated \
    --carbon-optimization balanced
```

#### Relaxed
- **Carbon Weight**: 20%
- **Quality Weight**: 60%
- **Max Model**: 405B
- **Use Case**: Priorità qualità

```bash
uv run python main.py demo --router-type integrated \
    --carbon-optimization relaxed \
    --enable-large-models
```

### Carbon Budget Management

```python
from src.routerllm.optimization import CarbonOptimizer

# Create optimizer with budgets
optimizer = CarbonOptimizer(
    daily_budget_kg=1.0,
    hourly_budget_kg=0.05,
    per_request_budget_kg=0.001,
    optimization_level="balanced"
)

# Check budget status
budget = optimizer.get_current_budget()
print(f"Remaining Daily: {budget.remaining_daily_kg} kg")
print(f"Remaining Hourly: {budget.remaining_hourly_kg} kg")

# Record usage
optimizer.record_carbon_usage(0.002)

# Check if over budget
if optimizer.is_over_budget():
    print("⚠️  Carbon budget exceeded!")
```

### Carbon Prediction

```python
from src.routerllm.optimization import CarbonPredictor

predictor = CarbonPredictor()

# Predict emissions before execution
prediction = predictor.predict_carbon_emissions(
    model_name="llama3_70b",
    num_params=70e9,
    num_tokens=500,
    use_quantization=True,
    use_flash_attention=True
)

print(f"Estimated Carbon: {prediction['estimated_carbon_kg']} kg")
print(f"Estimated Energy: {prediction['estimated_energy_kwh']} kWh")
print(f"Model Tier: {prediction['model_tier']}")
```

---

## 🧪 Testing

### Run All Tests
```bash
# Run comprehensive test suite
cd tests
uv run python run_all_tests.py
```

### Run Individual Tests
```bash
# Test MoE Router
uv run python tests/test_moe_router.py

# Test RL Router
uv run python tests/test_rl_router.py

# Test Carbon Optimizer
uv run python tests/test_carbon_optimizer.py

# Test Integrated System
uv run python tests/test_integrated_system.py
```

---

## 📈 Examples

### Example 1: Comparison of Router Strategies

```python
strategies = [
    RouterStrategy.DUMMY,
    RouterStrategy.BERT,
    RouterStrategy.DYNAMIC_MOE,
    RouterStrategy.REINFORCEMENT_LEARNING,
    RouterStrategy.CARBON_AWARE
]

test_prompts = [
    "Write a Python function",
    "Explain quantum computing",
    "Hello world",
    "Implement distributed system"
]

for strategy in strategies:
    config = SystemConfig(router_strategy=strategy)
    system = IntegratedRouterLLMSystem(
        config_path="configs/default_config.yaml",
        system_config=config
    )
    system.initialize()

    print(f"\n{'='*60}")
    print(f"Testing {strategy.value} router")
    print(f"{'='*60}")

    for prompt in test_prompts:
        result = system.process_request(prompt)
        print(f"{prompt[:40]:40} -> {result['model_used']}")

    system.cleanup()
```

### Example 2: Carbon Optimization Impact

```python
# Without optimization
config_no_opt = SystemConfig(
    router_strategy=RouterStrategy.DUMMY,
    enable_carbon_optimization=False
)

# With optimization
config_with_opt = SystemConfig(
    router_strategy=RouterStrategy.CARBON_AWARE,
    enable_carbon_optimization=True,
    carbon_optimization_level="balanced"
)

for config, label in [(config_no_opt, "Without"), (config_with_opt, "With")]:
    system = IntegratedRouterLLMSystem(
        config_path="configs/default_config.yaml",
        system_config=config
    )
    system.initialize()

    # Process 100 requests
    for i in range(100):
        system.process_request(f"Request {i}")

    report = system.get_system_report()
    print(f"{label} Optimization:")
    print(f"  Total Carbon: {report['statistics']['total_carbon_kg']} kg")
    print(f"  Avg per Request: {report['statistics']['avg_carbon_per_request']} kg")

    system.cleanup()
```

---

## 🔧 Configuration

### Production Config Example

```yaml
# configs/production_config.yaml
models:
  router:
    model_name: "bert-base-uncased"
    num_classes: 4

  llms:
    - name: "codellama_13b"
      model_id: "codellama/CodeLlama-13b-hf"
      category: 0
      complexity_tier: "heavy"
      use_4bit: true

    - name: "mistral_7b"
      model_id: "mistralai/Mistral-7B-Instruct-v0.1"
      category: 2
      complexity_tier: "medium"
      use_4bit: true

    - name: "phi3_mini"
      model_id: "microsoft/Phi-3-mini-4k-instruct"
      category: 3
      complexity_tier: "light"
      use_4bit: false

  large_llm:  # For 100B+ models
    name: "llama3_405b"
    model_id: "meta-llama/Llama-3.1-405B-Instruct"
    use_4bit: true
    use_flash_attention: true
    device_map: "auto"

carbon_tracking:
  enabled: true
  optimize_routing: true
  carbon_weight: 0.35
  daily_budget_kg: 1.0
  hourly_budget_kg: 0.05
  per_request_budget_kg: 0.001
  optimization_level: "balanced"
```

---

## 📊 Performance Metrics

### Expected Results

| Metric | Direct 100B+ | RouterLLM Optimized | Improvement |
|--------|--------------|---------------------|-------------|
| **Accuracy** | 95% | 88% | -7% (acceptable) |
| **CO2 Emissions** | 0.005 kg/req | 0.0015 kg/req | **-70%** ✅ |
| **Cost** | $0.002/req | $0.0004/req | **-80%** ✅ |
| **Latency** | 500ms | 150ms | **-70%** ✅ |
| **Throughput** | 10 req/s | 40 req/s | **+300%** ✅ |

---

## 🎓 Best Practices

1. **Start with Dummy Router** for development/testing
2. **Use BERT Router** for production if you have training data
3. **Use MoE Router** for dynamic workloads with varying complexity
4. **Use RL Router** when you want continuous improvement
5. **Use Integrated System** for production with carbon optimization
6. **Enable Caching** for repeated queries
7. **Enable Batching** for high-throughput scenarios
8. **Monitor Carbon** to track environmental impact
9. **Set Realistic Budgets** based on your usage patterns
10. **Test Thoroughly** before production deployment

---

## 🐛 Troubleshooting

### Import Errors
```bash
# Ensure all imports work
uv run python -c "from src.routerllm.models import DynamicMoERouter; print('OK')"
uv run python -c "from src.routerllm.models import ReinforcementLearningRouter; print('OK')"
uv run python -c "from src.routerllm.optimization import CarbonOptimizer; print('OK')"
```

### Memory Issues
- Enable 4-bit quantization in config
- Reduce batch sizes
- Use smaller models for development
- Disable large model support if not needed

### Carbon Tracking Not Working
- Check CodeCarbon is installed: `pip install codecarbon`
- Verify country code in config
- Ensure carbon tracking is enabled

---

## 📚 Additional Resources

- **Main Documentation**: `README.md`
- **Usage Guide**: `USAGE.md`
- **Academic Overview**: `docs/ACADEMIC_OVERVIEW.md`
- **Router Methods**: `router_methods.md`
- **Implementation Details**: `TODO_IMPLEMENTATION_COMPLETE.md`

---

**Sistema RouterLLM - Version 2.0 con Integrazione Completa** 🚀
