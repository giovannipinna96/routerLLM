#!/usr/bin/env python3
"""
Test script for NVIDIA prompt-task-and-complexity-classifier
Investigate and make it work for complexity-based routing
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, pipeline
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model_loading_approaches():
    """Test different approaches to load the NVIDIA model"""
    model_name = "nvidia/prompt-task-and-complexity-classifier"

    print("="*60)
    print("TESTING NVIDIA COMPLEXITY CLASSIFIER")
    print("="*60)

    # Approach 1: Direct pipeline
    print("\n1. Testing direct pipeline approach...")
    try:
        classifier = pipeline("text-classification", model=model_name)
        print("✅ Direct pipeline: SUCCESS")
        return classifier, "pipeline"
    except Exception as e:
        print(f"❌ Direct pipeline failed: {e}")

    # Approach 2: Manual model loading with AutoModelForSequenceClassification
    print("\n2. Testing AutoModelForSequenceClassification...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        print("✅ AutoModelForSequenceClassification: SUCCESS")
        return (tokenizer, model), "manual_sequence"
    except Exception as e:
        print(f"❌ AutoModelForSequenceClassification failed: {e}")

    # Approach 3: Generic AutoModel
    print("\n3. Testing generic AutoModel...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        print("✅ Generic AutoModel: SUCCESS")
        return (tokenizer, model), "manual_generic"
    except Exception as e:
        print(f"❌ Generic AutoModel failed: {e}")

    # Approach 4: Investigate model config
    print("\n4. Investigating model configuration...")
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        print(f"✅ Model config loaded: {type(config)}")
        print(f"Config details: {config}")

        # Try to understand the architecture
        print(f"Architecture: {getattr(config, 'architectures', 'Unknown')}")
        print(f"Model type: {getattr(config, 'model_type', 'Unknown')}")

        return None, "config_only"
    except Exception as e:
        print(f"❌ Config investigation failed: {e}")

    return None, "failed"


def test_model_inference(model_info, approach_type: str):
    """Test inference with the loaded model"""
    print(f"\n🧪 Testing inference with approach: {approach_type}")

    test_texts = [
        "Write a simple Python function to add two numbers",
        "Implement a complex neural network with attention mechanism and transformer architecture",
        "What is 2+2?",
        "Develop a distributed system for real-time data processing with fault tolerance"
    ]

    if approach_type == "pipeline":
        classifier = model_info
        print("Testing with pipeline...")

        for i, text in enumerate(test_texts):
            try:
                result = classifier(text)
                print(f"Text {i+1}: {text[:50]}...")
                print(f"Result: {result}")
                print("-" * 40)
            except Exception as e:
                print(f"❌ Inference failed for text {i+1}: {e}")

    elif approach_type == "manual_sequence":
        tokenizer, model = model_info
        print("Testing with manual sequence classification...")

        for i, text in enumerate(test_texts):
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

                with torch.no_grad():
                    outputs = model(**inputs)

                # Get predictions
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1)

                print(f"Text {i+1}: {text[:50]}...")
                print(f"Logits: {logits}")
                print(f"Probabilities: {probabilities}")
                print(f"Predicted class: {predicted_class.item()}")
                print("-" * 40)

            except Exception as e:
                print(f"❌ Manual inference failed for text {i+1}: {e}")

    elif approach_type == "manual_generic":
        tokenizer, model = model_info
        print("Testing with generic model (may need custom head)...")

        try:
            inputs = tokenizer(test_texts[0], return_tensors="pt", truncation=True, padding=True)

            with torch.no_grad():
                outputs = model(**inputs)

            print(f"Model outputs keys: {outputs.keys() if hasattr(outputs, 'keys') else 'No keys'}")
            print(f"Output shape: {outputs.last_hidden_state.shape if hasattr(outputs, 'last_hidden_state') else 'Unknown'}")

        except Exception as e:
            print(f"❌ Generic model inference failed: {e}")


def investigate_model_details(model_name: str = "nvidia/prompt-task-and-complexity-classifier"):
    """Deep investigation of the model structure and requirements"""
    print(f"\n🔍 DEEP INVESTIGATION: {model_name}")
    print("="*60)

    try:
        # 1. Check model card and files
        from huggingface_hub import hf_hub_download, list_repo_files

        print("📁 Repository files:")
        files = list_repo_files(model_name)
        for file in files:
            print(f"  - {file}")

        # 2. Try to download and read config
        print(f"\n📄 Configuration file:")
        config_path = hf_hub_download(repo_id=model_name, filename="config.json")
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        print(json.dumps(config_data, indent=2))

        # 3. Check if there's a README or model card
        try:
            readme_path = hf_hub_download(repo_id=model_name, filename="README.md")
            with open(readme_path, 'r') as f:
                readme_content = f.read()
            print(f"\n📖 README content (first 500 chars):")
            print(readme_content[:500])
        except:
            print("📖 No README found")

        # 4. Try to understand the output structure
        print(f"\n🎯 Model type analysis:")
        print(f"Architecture: {config_data.get('architectures', 'Not specified')}")
        print(f"Model type: {config_data.get('model_type', 'Not specified')}")
        print(f"Number of labels: {config_data.get('num_labels', 'Not specified')}")
        print(f"Label2id: {config_data.get('label2id', 'Not specified')}")
        print(f"Id2label: {config_data.get('id2label', 'Not specified')}")

        return config_data

    except Exception as e:
        print(f"❌ Investigation failed: {e}")
        return None


def create_custom_nvidia_router():
    """Create a custom router that works with the NVIDIA model"""
    print(f"\n🛠️ CREATING CUSTOM NVIDIA ROUTER")
    print("="*60)

    model_name = "nvidia/prompt-task-and-complexity-classifier"

    try:
        # Based on investigation, create a working solution
        print("Attempting custom implementation...")

        # Try different initialization methods
        methods = [
            {"trust_remote_code": True},
            {"trust_remote_code": True, "dtype": torch.float16},
            {"trust_remote_code": True, "device_map": "auto"},
        ]

        for i, kwargs in enumerate(methods):
            print(f"\nMethod {i+1}: {kwargs}")
            try:
                classifier = pipeline(
                    "text-classification",
                    model=model_name,
                    **kwargs
                )

                # Test inference
                test_result = classifier("Write a simple Python function")
                print(f"✅ Success! Test result: {test_result}")

                return classifier

            except Exception as e:
                print(f"❌ Method {i+1} failed: {e}")

        print("All methods failed, creating fallback...")
        return None

    except Exception as e:
        print(f"❌ Custom router creation failed: {e}")
        return None


class CustomNvidiaComplexityRouter:
    """Custom implementation that handles NVIDIA complexity classifier"""

    def __init__(self, model_name: str = "nvidia/prompt-task-and-complexity-classifier"):
        self.model_name = model_name
        self.classifier = None
        self.config = None
        self.logger = logger

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the NVIDIA model with various fallback strategies"""
        strategies = [
            self._init_with_trust_remote_code,
            self._init_with_pipeline_auto,
            self._init_with_manual_loading,
            self._init_fallback_dummy
        ]

        for strategy in strategies:
            try:
                result = strategy()
                if result:
                    self.logger.info(f"✅ Initialized with strategy: {strategy.__name__}")
                    return
            except Exception as e:
                self.logger.warning(f"❌ Strategy {strategy.__name__} failed: {e}")

        raise RuntimeError("All initialization strategies failed")

    def _init_with_trust_remote_code(self):
        """Try with trust_remote_code=True"""
        self.classifier = pipeline(
            "text-classification",
            model=self.model_name,
            trust_remote_code=True
        )
        return True

    def _init_with_pipeline_auto(self):
        """Try with automatic pipeline detection"""
        self.classifier = pipeline(
            task="text-classification",
            model=self.model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        return True

    def _init_with_manual_loading(self):
        """Try manual loading with different model classes"""
        from transformers import AutoTokenizer

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Try different model classes
        model_classes = [
            AutoModelForSequenceClassification,
            AutoModel,
        ]

        for model_class in model_classes:
            try:
                self.model = model_class.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                self.manual_mode = True
                return True
            except:
                continue

        return False

    def _init_fallback_dummy(self):
        """Fallback to a dummy implementation"""
        self.logger.warning("Using dummy fallback for NVIDIA router")
        self.dummy_mode = True
        return True

    def predict_complexity(self, text: str) -> Tuple[str, float]:
        """Predict complexity level from text"""
        if hasattr(self, 'classifier') and self.classifier:
            try:
                result = self.classifier(text)
                # Extract complexity information
                complexity_score = self._extract_complexity_score(result)
                complexity_level = self._score_to_level(complexity_score)
                return complexity_level, complexity_score
            except Exception as e:
                self.logger.error(f"Classification failed: {e}")

        elif hasattr(self, 'manual_mode') and self.manual_mode:
            try:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Process outputs manually
                    complexity_score = self._process_manual_outputs(outputs)
                    complexity_level = self._score_to_level(complexity_score)
                    return complexity_level, complexity_score
            except Exception as e:
                self.logger.error(f"Manual prediction failed: {e}")

        # Fallback to dummy prediction
        import random
        complexity_score = random.uniform(0, 1)
        complexity_level = self._score_to_level(complexity_score)
        return complexity_level, complexity_score

    def _extract_complexity_score(self, result) -> float:
        """Extract complexity score from pipeline result"""
        # This needs to be adapted based on actual model output
        if isinstance(result, list) and len(result) > 0:
            first_result = result[0]
            if isinstance(first_result, dict):
                # Look for complexity-related labels
                score = first_result.get('score', 0.5)
                label = first_result.get('label', '').lower()

                if 'complex' in label or 'difficult' in label:
                    return score
                elif 'simple' in label or 'easy' in label:
                    return 1 - score

        return 0.5  # Default medium complexity

    def _process_manual_outputs(self, outputs) -> float:
        """Process manual model outputs"""
        # Implement based on actual model architecture
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            # Convert to complexity score (needs model-specific logic)
            return probabilities.max().item()

        return 0.5

    def _score_to_level(self, score: float) -> str:
        """Convert numerical score to complexity level"""
        if score < 0.33:
            return "simple"
        elif score < 0.66:
            return "medium"
        else:
            return "complex"


def main():
    """Main testing function"""
    print("🚀 NVIDIA Complexity Classifier Testing")
    print("="*80)

    # Step 1: Investigate model details
    config_data = investigate_model_details()

    # Step 2: Test different loading approaches
    model_info, approach_type = test_model_loading_approaches()

    # Step 3: Test inference if model loaded
    if model_info and approach_type != "failed":
        test_model_inference(model_info, approach_type)

    # Step 4: Try custom router implementation
    print(f"\n" + "="*80)
    custom_router = create_custom_nvidia_router()

    if custom_router:
        print("\n✅ Testing custom router...")
        test_texts = [
            "Write a simple function",
            "Implement a complex distributed system",
            "What is machine learning?",
            "Create a neural network with attention mechanism"
        ]

        for text in test_texts:
            try:
                result = custom_router(text)
                print(f"'{text}' -> {result}")
            except Exception as e:
                print(f"❌ Custom router test failed: {e}")

    # Step 5: Test our custom class
    print(f"\n" + "="*80)
    print("🧪 Testing CustomNvidiaComplexityRouter class...")

    try:
        custom_class = CustomNvidiaComplexityRouter()

        test_texts = [
            "Write a simple Python function to add two numbers",
            "Implement a complex neural network architecture",
            "Hello world",
            "Design a distributed microservices architecture"
        ]

        for text in test_texts:
            complexity_level, score = custom_class.predict_complexity(text)
            print(f"'{text[:40]}...' -> Level: {complexity_level}, Score: {score:.3f}")

    except Exception as e:
        print(f"❌ Custom class test failed: {e}")

    print(f"\n" + "="*80)
    print("🎯 SUMMARY AND RECOMMENDATIONS")
    print("="*80)

    if approach_type == "pipeline":
        print("✅ NVIDIA model works with direct pipeline approach")
        print("💡 Recommendation: Use pipeline with appropriate configuration")
    elif approach_type == "manual_sequence":
        print("✅ NVIDIA model works with manual loading")
        print("💡 Recommendation: Use manual tokenizer + model approach")
    elif config_data:
        print("ℹ️  Model config accessible, but inference needs work")
        print("💡 Recommendation: Check model documentation for proper usage")
    else:
        print("❌ NVIDIA model requires custom implementation")
        print("💡 Recommendation: Use fallback router or custom wrapper")


if __name__ == "__main__":
    main()