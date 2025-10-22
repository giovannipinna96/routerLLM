# RouterLLM - Guida Completa all'Uso

## Panoramica del Sistema

RouterLLM è un sistema intelligente di routing che seleziona automaticamente il Large Language Model (LLM) più appropriato per ogni richiesta, ottimizzando il rapporto tra accuratezza, costi computazionali e consumo energetico.

## Architettura del Sistema

### Componenti Principali

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Text    │───▶│  Router (BERT)   │───▶│  LLM Selection  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Carbon Tracker │◀───│  LLM Manager     │───▶│   Response      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

#### 1. **Router (Cervello del Sistema)**
- **Router BERT**: Classificatore basato su BERT che analizza il testo di input e predice la categoria
- **Router Dummy**: Versione di test che seleziona casualmente (utile per sviluppo)

#### 2. **LLM Manager (Gestore dei Modelli)**
- Gestisce il caricamento/scaricamento automatico dei modelli
- Ottimizza l'uso della memoria GPU con quantizzazione 4-bit
- Supporta modelli fino a 15B parametri

#### 3. **Carbon Tracker (Monitoraggio Ambientale)**
- Traccia le emissioni di CO2 in tempo reale usando CodeCarbon
- Monitora separatamente: inferenza router, caricamento modelli, generazione

#### 4. **Sistema di Training**
- Training standard con cross-entropy loss
- **Inter-Intra Loss**: Loss personalizzata per migliorare la separazione tra classi

## Categorie di LLM

Il sistema classifica le richieste in 4 categorie:

| Categoria | Modello | Specializzazione | Esempi |
|-----------|---------|------------------|--------|
| **0 - Code Generation** | CodeLlama-7B* | Programmazione, debug, algoritmi | "Scrivi una funzione Python", "Implementa un algoritmo" |
| **1 - Text Generation** | Phi-3-Mini* | Scrittura creativa, contenuti | "Scrivi una storia", "Crea un articolo" |
| **2 - General Purpose** | Mistral-7B* | Analisi, spiegazioni, consigli | "Spiega la differenza tra...", "Vantaggi di..." |
| **3 - Lightweight Tasks** | TinyLlama-1.1B | Domande semplici, traduzioni | "Cos'è l'AI?", "Traduci in spagnolo" |

*Nota: Nella configurazione di test attuale tutti utilizzano TinyLlama per velocità*

## Installazione e Setup

### 1. Preparazione Ambiente
```bash
# Clona il repository
cd routerLLM

# Le dipendenze sono già configurate in pyproject.toml
# uv installerà automaticamente tutto al primo utilizzo
```

### 2. Configurazione
Il sistema è configurato tramite `configs/default_config.yaml`:

```yaml
models:
  router:
    model_name: "bert-base-uncased"
    num_classes: 4
    max_length: 512

  llms:
    - name: "tiny_llama"
      model_id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
      category: 0
    # ... altri modelli

carbon_tracking:
  enabled: true
  country_iso_code: "USA"
  output_dir: "./logs/carbon"
```

## Utilizzo del Sistema

### 1. Generazione Dataset di Training

Prima di tutto, genera i dati sintetici per il training del router:

```bash
# Genera 1200 campioni (default)
uv run python main.py generate-data --samples 1200 --output-dir ./data

# Per test rapidi
uv run python main.py generate-data --samples 100 --output-dir ./test_data
```

**Output:**
```
Dataset generated and saved to ./data
Train samples: 840
Validation samples: 180
Test samples: 180
```

### 2. Training del Router BERT

Addestra il router per la classificazione intelligente:

```bash
# Training standard
uv run python main.py train --data-dir ./data --model-dir ./models --epochs 5

# Training con Inter-Intra Loss
uv run python main.py train --data-dir ./data --use-inter-intra-loss --epochs 5

# Training veloce per test
uv run python main.py train --data-dir ./test_data --epochs 1 --batch-size 8
```

**Output del Training:**
```
Training BERT router...
Epoch 1/5
Train Loss: 1.2456, Train Accuracy: 0.6750
Val Loss: 1.1234, Val Accuracy: 0.7222
New best model saved: ./models/best_router_epoch_1.pt
Training completed!
```

### 3. Test del Sistema

#### Test Rapido con Router Dummy
```bash
uv run python main.py test --router-type dummy
```

#### Test con Router BERT Addestrato
```bash
uv run python main.py test --router-type bert --router-model ./models/best_router_epoch_1.pt
```

#### Test con Esempi Predefiniti
```bash
uv run python main.py test --router-type bert --router-model ./models/best_router_epoch_1.pt --test-examples
```

### 4. Modalità Demo Interattiva

Lancia il sistema in modalità interattiva:

```bash
# Con router dummy
uv run python main.py demo --router-type dummy

# Con router addestrato
uv run python main.py demo --router-type bert --router-model ./models/best_router_epoch_1.pt
```

**Esempio di Sessione Interattiva:**
```
ROUTERLLM INTERACTIVE DEMO
============================================================
Router Type: bert
Available LLMs: tiny_llama, tiny_llama_text, tiny_llama_general, tiny_llama_light

Enter your prompt: Scrivi una funzione Python per calcolare il fattoriale

Processing...
Predicted Model: tiny_llama
Confidence: 0.8456
Response: def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)
Total Time: 2.3456s
```

## Utilizzo Programmatico

### Esempio Base
```python
from src.routerllm.core.system import RouterLLMSystem

# Inizializza il sistema
system = RouterLLMSystem(
    config_path="configs/default_config.yaml",
    router_type="dummy",  # o "bert"
    router_model_path="./models/best_router.pt",  # solo per BERT
    enable_carbon_tracking=True
)

# Inizializza tutti i componenti
system.initialize()

# Genera una risposta
result = system.predict_and_generate(
    "Scrivi una funzione Python per ordinare una lista",
    max_length=256,
    temperature=0.7
)

# Mostra i risultati
print(f"Modello selezionato: {result['predicted_model']}")
print(f"Confidenza: {result['confidence']:.4f}")
print(f"Risposta: {result['response']}")
print(f"Tempo totale: {result['timing']['total_time']:.4f}s")

# Cleanup
system.cleanup()
```

### Elaborazione in Batch
```python
# Lista di prompts
prompts = [
    "Implementa un algoritmo di ordinamento",
    "Spiega il machine learning",
    "Cos'è l'intelligenza artificiale?",
    "Scrivi una storia su un robot"
]

# Elabora tutto in batch
results = system.batch_process(prompts, max_length=200)

# Analizza i risultati
for i, result in enumerate(results):
    print(f"Prompt {i+1}: {result['predicted_model']}")
```

### Cambio Router in Runtime
```python
# Inizia con dummy router
system = RouterLLMSystem(router_type="dummy")
system.initialize()

# Cambia a router BERT
system.switch_router("bert", "./models/best_router.pt")

# Testa entrambi
result1 = system.predict_and_generate("Test prompt")
```

## Monitoraggio e Analisi

### 1. Statistiche del Sistema
```python
stats = system.get_system_stats()
print(f"Richieste totali: {stats['total_requests']}")
print(f"Tempo medio inferenza: {stats['average_inference_time']:.4f}s")
print(f"Emissioni totali CO2: {stats['carbon_footprint']['total_emissions_kg']:.6f} kg")
```

### 2. Carbon Footprint Dettagliato
Le emissioni sono tracciate per componente:

```
CARBON FOOTPRINT BREAKDOWN:
router_inference: 0.000005 kg CO2 (1.7%)
model_loading: 0.000253 kg CO2 (84.2%)
llm_inference: 0.000042 kg CO2 (14.1%)
TOTAL: 0.000300 kg CO2
Equivalent to driving ~0.00 km in a car
```

### 3. File di Log
Il sistema genera log dettagliati in `./logs/`:

- `routerllm.log`: Log generale del sistema
- `carbon/emissions.csv`: Dati delle emissioni
- `router_training.log`: Log del training

## Configurazione Avanzata

### Personalizzazione Modelli
Modifica `configs/default_config.yaml` per usare modelli diversi:

```yaml
llms:
  - name: "codellama_7b"
    model_id: "codellama/CodeLlama-7b-hf"
    max_memory: "8GB"
    use_4bit: true
    category: 0

  - name: "mistral_7b"
    model_id: "mistralai/Mistral-7B-Instruct-v0.1"
    max_memory: "8GB"
    use_4bit: true
    category: 2
```

### Ottimizzazione Memoria
Per sistemi con memoria limitata:

```yaml
system:
  max_gpu_memory: "8GB"
  torch_dtype: "float16"

# Abilita quantizzazione 4-bit
llms:
  - name: "model_name"
    use_4bit: true
```

### Training Personalizzato
```python
from src.routerllm.training.losses import InterIntraLoss

# Usa Inter-Intra Loss per migliorare la separazione tra classi
loss_fn = InterIntraLoss(
    num_classes=4,
    alpha=1.0,    # peso classification loss
    beta=0.5,     # peso inter-class loss
    gamma=0.3,    # peso intra-class loss
    margin=1.0    # margine per separazione inter-class
)
```

## Troubleshooting

### Problemi Comuni

1. **Errore di Memoria GPU**
   ```bash
   # Riduci batch size
   uv run python main.py train --batch-size 4

   # Abilita quantizzazione
   # Modifica config: use_4bit: true
   ```

2. **Modelli Non Si Caricano**
   ```bash
   # Verifica connessione internet per download da Hugging Face
   # Controlla spazio disco per cache modelli
   ls ~/.cache/huggingface/
   ```

3. **Carbon Tracking Non Funziona**
   ```bash
   # Installa codecarbon esplicitamente
   pip install codecarbon

   # Verifica configurazione paese
   # Modifica config: country_iso_code: "ITA"
   ```

### Ottimizzazione Performance

1. **Per Training Veloce**
   - Usa dataset piccoli (100-500 campioni)
   - Riduci epochs (1-2)
   - Aumenta batch_size se hai memoria

2. **Per Inferenza Veloce**
   - Usa router dummy per test
   - Mantieni modelli caricati tra richieste
   - Abilita quantizzazione 4-bit

3. **Per Ridurre Emissioni**
   - Usa modelli più piccoli (TinyLlama)
   - Batch multiple richieste insieme
   - Evita caricamenti frequenti di modelli

## Esempi Pratici

### Caso d'Uso 1: Assistente Programmazione
```python
system = RouterLLMSystem(router_type="bert")
system.initialize()

code_requests = [
    "Implementa quicksort in Python",
    "Come debuggare un segmentation fault?",
    "Scrivi test unitari per questa funzione",
    "Ottimizza questo algoritmo di ricerca"
]

for request in code_requests:
    result = system.predict_and_generate(request)
    print(f"→ {result['predicted_model']}: {result['response'][:100]}...")
```

### Caso d'Uso 2: Content Generation
```python
content_requests = [
    "Scrivi un articolo sui benefici del remote work",
    "Crea una storia per bambini su un robot gentile",
    "Descrivi un prodotto tech innovativo",
    "Scrivi una recensione di un libro sci-fi"
]

# Il router dovrebbe selezionare modelli specializzati per testo
results = system.batch_process(content_requests)
```

### Caso d'Uso 3: Analisi e Spiegazioni
```python
analysis_requests = [
    "Qual è la differenza tra ML e AI?",
    "Vantaggi e svantaggi del cloud computing",
    "Come funziona la blockchain?",
    "Impatto dell'AI sulla società"
]

# Router dovrebbe usare modelli general-purpose
results = system.batch_process(analysis_requests)
```

## Prossimi Sviluppi

Il sistema è progettato per essere facilmente estendibile:

1. **Nuovi Modelli**: Aggiungi semplicemente nuove configurazioni
2. **Nuove Categorie**: Espandi il router per più classi
3. **Metriche Custom**: Aggiungi nuovi tracker per monitoraggio
4. **Loss Functions**: Implementa nuove funzioni di loss per training

---

**Nota**: Questa guida copre tutti gli aspetti del sistema RouterLLM. Per domande specifiche o problemi, consulta i log del sistema o contatta il team di sviluppo.