"""
Training utilities for BERT Router
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import get_linear_schedule_with_warmup

from ..models.router import BERTRouter


class RouterDataset(Dataset):
    """Dataset class for router training"""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 512
    ):
        """
        Initialize dataset

        Args:
            texts: List of input texts
            labels: List of corresponding labels
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class RouterTrainer:
    """
    Trainer for BERT Router
    """

    def __init__(
        self,
        router: BERTRouter,
        train_dataset: RouterDataset,
        val_dataset: Optional[RouterDataset] = None,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        save_dir: str = "./models",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize trainer

        Args:
            router: BERTRouter instance
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            batch_size: Batch size for training
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay for optimizer
            gradient_accumulation_steps: Gradient accumulation steps
            save_dir: Directory to save models
            logger: Logger instance
        """
        self.router = router
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.save_dir = save_dir
        self.logger = logger or logging.getLogger(__name__)

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Initialize model if not already done
        if self.router.model is None:
            self.router._build_model()

        # Setup training components
        self._setup_training()

        self.logger.info(f"RouterTrainer initialized - Train samples: {len(train_dataset)}")
        if val_dataset:
            self.logger.info(f"Validation samples: {len(val_dataset)}")

    def _setup_training(self):
        """Setup training components"""
        # Data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2
            )

        # Optimizer
        self.optimizer = optim.AdamW(
            self.router.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Scheduler
        total_steps = len(self.train_loader) * self.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        self.logger.info(f"Training setup complete - Total steps: {total_steps}")

    def train(self) -> Dict[str, Any]:
        """
        Train the router model

        Returns:
            Training history dictionary
        """
        self.logger.info("Starting training...")

        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

        best_val_accuracy = 0.0
        best_model_path = None

        for epoch in range(self.num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")

            # Training phase
            train_loss, train_accuracy = self._train_epoch()
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_accuracy)

            self.logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

            # Validation phase
            if self.val_dataset:
                val_loss, val_accuracy = self._validate_epoch()
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)

                self.logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

                # Save best model
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_model_path = os.path.join(self.save_dir, f"best_router_epoch_{epoch + 1}.pt")
                    self.router.save_model(
                        best_model_path,
                        self.optimizer,
                        epoch + 1,
                        val_loss
                    )
                    self.logger.info(f"New best model saved: {best_model_path}")

            # Save checkpoint
            checkpoint_path = os.path.join(self.save_dir, f"router_epoch_{epoch + 1}.pt")
            self.router.save_model(
                checkpoint_path,
                self.optimizer,
                epoch + 1,
                train_loss
            )

        self.logger.info("Training completed!")
        if best_model_path:
            self.logger.info(f"Best model: {best_model_path} (Val Accuracy: {best_val_accuracy:.4f})")

        return history

    def _train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.router.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.router.device)
            attention_mask = batch['attention_mask'].to(self.router.device)
            labels = batch['label'].to(self.router.device)

            # Forward pass
            outputs = self.router.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = self.criterion(outputs, labels)

            # Backward pass
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            loss.backward()

            # Update weights
            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Track metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            predictions = torch.argmax(outputs, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)

        return avg_loss, accuracy

    def _validate_epoch(self) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.router.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.router.device)
                attention_mask = batch['attention_mask'].to(self.router.device)
                labels = batch['label'].to(self.router.device)

                # Forward pass
                outputs = self.router.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs, labels)

                # Track metrics
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate epoch metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)

        return avg_loss, accuracy

    def evaluate(self, test_dataset: RouterDataset) -> Dict[str, Any]:
        """
        Evaluate the model on test dataset

        Args:
            test_dataset: Test dataset

        Returns:
            Evaluation metrics
        """
        self.logger.info("Starting evaluation...")

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

        self.router.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.router.device)
                attention_mask = batch['attention_mask'].to(self.router.device)
                labels = batch['label'].to(self.router.device)

                # Forward pass
                outputs = self.router.model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.softmax(outputs, dim=-1)
                predictions = torch.argmax(outputs, dim=-1)

                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        conf_matrix = confusion_matrix(all_labels, all_predictions)

        # Calculate per-class metrics
        per_class_metrics = precision_recall_fscore_support(
            all_labels, all_predictions, average=None
        )

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix.tolist(),
            'per_class_precision': per_class_metrics[0].tolist(),
            'per_class_recall': per_class_metrics[1].tolist(),
            'per_class_f1': per_class_metrics[2].tolist(),
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'labels': all_labels
        }

        self.logger.info(f"Evaluation completed - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        return results

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'warmup_steps': self.warmup_steps,
            'weight_decay': self.weight_decay,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'train_samples': len(self.train_dataset),
            'val_samples': len(self.val_dataset) if self.val_dataset else 0
        }