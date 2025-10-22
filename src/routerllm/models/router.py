"""
Router models for LLM selection
"""

import torch
import torch.nn as nn
import numpy as np
import random
import logging
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModel, pipeline
from abc import ABC, abstractmethod


class BaseRouter(ABC):
    """Base class for all router implementations"""

    @abstractmethod
    def predict(self, text: str) -> Tuple[int, float]:
        """
        Predict the best LLM for the given text

        Args:
            text: Input text

        Returns:
            Tuple of (predicted_class, confidence_score)
        """
        pass

    @abstractmethod
    def get_model_name_from_class(self, class_id: int) -> str:
        """
        Get model name from class ID

        Args:
            class_id: Predicted class ID

        Returns:
            Model name
        """
        pass


class BERTRouter(BaseRouter):
    """
    BERT-based router for LLM selection
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 4,
        max_length: int = 512,
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize BERT Router

        Args:
            model_name: Pre-trained BERT model name
            num_classes: Number of LLM classes
            max_length: Maximum sequence length
            device: Device to run on
            logger: Logger instance
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        self.logger = logger or logging.getLogger(__name__)

        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None

        # Class to model name mapping (should match config)
        self.class_to_model = {
            0: "codellama_13b",     # Complex tasks -> Large model (13B)
            1: "mistral_7b",        # Medium tasks -> Medium model (7B)
            2: "codellama_7b",      # General tasks -> Medium model (7B)
            3: "phi3_mini"          # Simple tasks -> Small model (3.8B)
        }

        self.logger.info(f"BERTRouter initialized with {num_classes} classes on {self.device}")

    def _build_model(self):
        """Build the BERT classification model"""
        try:
            # Load pre-trained BERT
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            bert_model = AutoModel.from_pretrained(self.model_name)

            # Create classification head
            self.model = BERTClassifier(
                bert_model=bert_model,
                num_classes=self.num_classes,
                dropout_rate=0.1
            )

            self.model.to(self.device)
            self.logger.info(f"BERT model built successfully: {self.model_name}")

        except Exception as e:
            self.logger.error(f"Failed to build BERT model: {e}")
            raise

    def load_model(self, model_path: str):
        """
        Load a trained model from file

        Args:
            model_path: Path to the saved model
        """
        try:
            if self.model is None:
                self._build_model()

            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            self.logger.info(f"Model loaded from {model_path}")

        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def save_model(self, model_path: str, optimizer=None, epoch: int = 0, loss: float = 0.0):
        """
        Save the current model

        Args:
            model_path: Path to save the model
            optimizer: Optimizer state (optional)
            epoch: Current epoch
            loss: Current loss
        """
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'model_name': self.model_name,
                'num_classes': self.num_classes,
                'max_length': self.max_length,
                'epoch': epoch,
                'loss': loss,
            }

            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()

            torch.save(checkpoint, model_path)
            self.logger.info(f"Model saved to {model_path}")

        except Exception as e:
            self.logger.error(f"Failed to save model to {model_path}: {e}")
            raise

    def predict(self, text: str) -> Tuple[int, float]:
        """
        Predict the best LLM for the given text

        Args:
            text: Input text

        Returns:
            Tuple of (predicted_class, confidence_score)
        """
        if self.model is None:
            self.logger.error("Model not loaded. Please load a trained model first.")
            raise ValueError("Model not loaded")

        try:
            self.model.eval()

            # Tokenize input
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )

            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            # Predict
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.softmax(outputs, dim=-1)
                confidence, predicted_class = torch.max(probabilities, dim=-1)

            predicted_class = predicted_class.item()
            confidence_score = confidence.item()

            self.logger.debug(f"Prediction: class={predicted_class}, confidence={confidence_score:.4f}")

            return predicted_class, confidence_score

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise

    def predict_batch(self, texts: List[str]) -> List[Tuple[int, float]]:
        """
        Predict for a batch of texts

        Args:
            texts: List of input texts

        Returns:
            List of (predicted_class, confidence_score) tuples
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        try:
            self.model.eval()
            results = []

            # Process in batches
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # Tokenize batch
                encoding = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )

                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)

                # Predict batch
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    probabilities = torch.softmax(outputs, dim=-1)
                    confidences, predicted_classes = torch.max(probabilities, dim=-1)

                # Collect results
                for pred_class, confidence in zip(predicted_classes.cpu().numpy(), confidences.cpu().numpy()):
                    results.append((int(pred_class), float(confidence)))

            return results

        except Exception as e:
            self.logger.error(f"Batch prediction failed: {e}")
            raise

    def get_model_name_from_class(self, class_id: int) -> str:
        """Get model name from class ID"""
        return self.class_to_model.get(class_id, "unknown")

    def get_model_config(self):
        """Get model configuration"""
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'max_length': self.max_length,
            'device': str(self.device),
            'class_to_model': self.class_to_model
        }


class BERTClassifier(nn.Module):
    """
    BERT-based classifier for router
    """

    def __init__(self, bert_model, num_classes: int, dropout_rate: float = 0.1):
        """
        Initialize BERT classifier

        Args:
            bert_model: Pre-trained BERT model
            num_classes: Number of output classes
            dropout_rate: Dropout rate
        """
        super(BERTClassifier, self).__init__()

        self.bert = bert_model
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Classification logits
        """
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        pooled_output = outputs.pooler_output

        # Apply dropout and classify
        output = self.dropout(pooled_output)
        logits = self.classifier(output)

        return logits


class DummyRouter(BaseRouter):
    """
    Dummy router that randomly selects LLMs for testing
    """

    def __init__(
        self,
        num_classes: int = 4,
        seed: Optional[int] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Dummy Router

        Args:
            num_classes: Number of LLM classes
            seed: Random seed for reproducibility
            logger: Logger instance
        """
        self.num_classes = num_classes
        self.logger = logger or logging.getLogger(__name__)

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Class to model name mapping (should match config)
        self.class_to_model = {
            0: "codellama_13b",     # Complex tasks -> Large model (13B)
            1: "mistral_7b",        # Medium tasks -> Medium model (7B)
            2: "codellama_7b",      # General tasks -> Medium model (7B)
            3: "phi3_mini"          # Simple tasks -> Small model (3.8B)
        }

        self.logger.info(f"DummyRouter initialized with {num_classes} classes")

    def predict(self, text: str) -> Tuple[int, float]:
        """
        Randomly predict an LLM class

        Args:
            text: Input text (ignored)

        Returns:
            Tuple of (predicted_class, confidence_score)
        """
        # Random prediction
        predicted_class = random.randint(0, self.num_classes - 1)
        confidence_score = random.uniform(0.5, 1.0)  # Random confidence

        self.logger.debug(f"Dummy prediction: class={predicted_class}, confidence={confidence_score:.4f}")

        return predicted_class, confidence_score

    def predict_batch(self, texts: List[str]) -> List[Tuple[int, float]]:
        """
        Predict for a batch of texts

        Args:
            texts: List of input texts

        Returns:
            List of (predicted_class, confidence_score) tuples
        """
        return [self.predict(text) for text in texts]

    def get_model_name_from_class(self, class_id: int) -> str:
        """Get model name from class ID"""
        return self.class_to_model.get(class_id, "unknown")

    def set_seed(self, seed: int):
        """Set random seed"""
        random.seed(seed)
        np.random.seed(seed)
        self.logger.info(f"Random seed set to {seed}")

    def get_model_config(self):
        """Get model configuration"""
        return {
            'router_type': 'dummy',
            'num_classes': self.num_classes,
            'class_to_model': self.class_to_model
        }


class GrahamComplexityRouter(BaseRouter):
    """
    Complexity-based router using grahamaco/question-complexity-classifier
    Routes based on complexity classification to models of different sizes
    """

    def __init__(
        self,
        model_name: str = "grahamaco/question-complexity-classifier",
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Graham Complexity Router

        Args:
            model_name: Hugging Face model name for complexity classification
            device: Device to run on
            logger: Logger instance
        """
        self.model_name = model_name
        self.logger = logger or logging.getLogger(__name__)

        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize the pipeline
        try:
            self.classifier = pipeline(
                "text-classification",
                model=self.model_name,
                device=0 if self.device.type == "cuda" else -1
            )
            self.logger.info(f"GrahamComplexityRouter initialized with {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Graham complexity classifier: {e}")
            raise

        # Define complexity label mappings for Graham model
        # Using only CodeLlama models (working models)
        self.class_to_model = {
            0: "codellama_13b",     # Hard tasks -> Large model (13B)
            1: "mistral_7b",        # Medium tasks -> Medium model (7B)
            2: "codellama_7b",      # Medium tasks -> Medium model (7B)
            3: "codellama_7b"       # Easy tasks -> Medium model (7B, avoid problematic phi3)
        }

        # Graham complexity mapping (uses LABEL_0, LABEL_1)
        self.complexity_to_class = {
            "LABEL_0": 2,     # Simple/Easy -> CodeLlama-7B (reliable)
            "LABEL_1": 0,     # Complex/Hard -> CodeLlama-13B
        }

    def predict(self, text: str) -> Tuple[int, float]:
        """
        Predict the best LLM based on complexity score

        Args:
            text: Input text

        Returns:
            Tuple of (predicted_class, complexity_score)
        """
        try:
            # Get complexity prediction from Graham model
            results = self.classifier(text)

            # Extract the highest scoring label from Graham model
            # Expected output: [{'label': 'LABEL_0'/'LABEL_1', 'score': float}]
            if isinstance(results, list) and len(results) > 0:
                top_result = results[0]
                complexity_label = top_result['label']  # Keep original case
                confidence_score = top_result['score']
            else:
                # Fallback
                complexity_label = "LABEL_1"  # Default to complex
                confidence_score = 0.5

            # Map Graham complexity label to model class
            predicted_class = self.complexity_to_class.get(complexity_label, 2)

            self.logger.debug(f"Graham Complexity prediction: label={complexity_label}, "
                            f"confidence={confidence_score:.4f}, class={predicted_class}")

            return predicted_class, confidence_score

        except Exception as e:
            self.logger.error(f"Graham complexity prediction failed: {e}")
            # Fallback to medium complexity
            return 2, 0.5

    def get_model_name_from_class(self, class_id: int) -> str:
        """Get model name from class ID"""
        class_to_model = {
            0: "codellama_13b",     # Complex tasks -> Large model (13B)
            1: "mistral_7b",        # Medium tasks -> Medium model (7B)
            2: "codellama_7b",      # General tasks -> Medium model (7B)
            3: "codellama_7b"       # Simple tasks -> Medium model (7B, avoid phi3)
        }
        return class_to_model.get(class_id, "mistral_7b")

    def get_model_config(self):
        """Get model configuration"""
        return {
            'router_type': 'graham_complexity',
            'model_name': self.model_name,
            'device': str(self.device),
            'class_to_model': self.class_to_model,
            'complexity_to_class': self.complexity_to_class
        }