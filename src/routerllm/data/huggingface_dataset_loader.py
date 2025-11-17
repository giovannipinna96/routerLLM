"""
HuggingFace dataset loader for router training
Loads datasets from HuggingFace Hub and converts them to RouterLLM format
"""

import json
import os
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from tqdm import tqdm


class HuggingFaceDatasetLoader:
    """
    Loader for HuggingFace datasets compatible with RouterLLM
    """

    def __init__(
        self,
        dataset_name: str = "nampdn-ai/tiny-codes",
        seed: int = 42,
        logger: logging.Logger = None
    ):
        """
        Initialize HuggingFace dataset loader

        Args:
            dataset_name: HuggingFace dataset identifier
            seed: Random seed for reproducibility
            logger: Logger instance
        """
        self.dataset_name = dataset_name
        self.seed = seed
        self.logger = logger or logging.getLogger(__name__)

        # Set random seeds
        np.random.seed(seed)

        # Define category mapping for tiny-codes dataset
        # Maps target_audience values to RouterLLM categories
        self.target_audience_mapping = {
            "Beginners": 3,      # Lightweight tasks
            "Experts": 2,        # General purpose / Medium
            "Professionals": 0,  # Heavy / Complex code generation
        }

        # Define categories (compatible with RouterLLM system)
        self.categories = {
            0: {
                "name": "code_generation_professional",
                "model": "codellama_13b",
                "description": "Professional-level code generation tasks",
                "complexity": "heavy"
            },
            2: {
                "name": "code_generation_expert",
                "model": "mistral_7b",
                "description": "Expert-level code tasks",
                "complexity": "medium"
            },
            3: {
                "name": "code_generation_beginner",
                "model": "phi3_mini",
                "description": "Beginner-level code tasks",
                "complexity": "light"
            }
        }
        # Note: Category 1 (text generation) is not populated for code-only datasets

        self.logger.info(f"HuggingFaceDatasetLoader initialized for dataset: {dataset_name}")

    def load_huggingface_dataset(
        self,
        max_samples: Optional[int] = None,
        split: str = "train"
    ) -> Tuple[List[str], List[int], List[str], List[str]]:
        """
        Load dataset from HuggingFace Hub

        Args:
            max_samples: Maximum number of samples to load (None = all)
            split: Dataset split to load (train, test, validation)

        Returns:
            Tuple of (texts, labels, scenarios, responses)
        """
        self.logger.info(f"Loading dataset '{self.dataset_name}' from HuggingFace Hub...")

        try:
            # Load dataset
            dataset = load_dataset(self.dataset_name, split=split)

            self.logger.info(f"Dataset loaded successfully. Total samples: {len(dataset)}")

            # Extract data
            texts = []
            labels = []
            scenarios = []
            responses = []

            skipped = 0

            # Process samples
            samples_to_process = len(dataset) if max_samples is None else min(max_samples, len(dataset))

            self.logger.info(f"Processing {samples_to_process} samples...")

            for i in tqdm(range(samples_to_process), desc="Loading samples"):
                sample = dataset[i]

                # Extract fields
                prompt = sample.get("prompt", None)
                target_audience = sample.get("target_audience", None)
                scenario = sample.get("scenario", "")
                response = sample.get("response", "")

                # Validate and map
                if prompt is None or target_audience is None:
                    skipped += 1
                    continue

                # Map target_audience to category
                if target_audience not in self.target_audience_mapping:
                    # Try case-insensitive match
                    target_audience_lower = {k.lower(): v for k, v in self.target_audience_mapping.items()}
                    target_audience_key = target_audience.lower() if isinstance(target_audience, str) else None

                    if target_audience_key and target_audience_key in target_audience_lower:
                        category = target_audience_lower[target_audience_key]
                    else:
                        skipped += 1
                        continue
                else:
                    category = self.target_audience_mapping[target_audience]

                # Add to lists
                texts.append(str(prompt))
                labels.append(category)
                scenarios.append(str(scenario) if scenario else "unknown")
                responses.append(str(response) if response else "")

            self.logger.info(f"Processed {len(texts)} samples successfully")
            if skipped > 0:
                self.logger.warning(f"Skipped {skipped} samples due to missing/invalid fields")

            return texts, labels, scenarios, responses

        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise

    def balance_classes(
        self,
        texts: List[str],
        labels: List[int],
        scenarios: List[str],
        responses: List[str]
    ) -> Tuple[List[str], List[int], List[str], List[str]]:
        """
        Balance classes to have equal representation

        Args:
            texts: List of texts
            labels: List of labels
            scenarios: List of scenarios
            responses: List of responses

        Returns:
            Tuple of balanced (texts, labels, scenarios, responses)
        """
        # Group by label
        label_groups = {}
        for i, label in enumerate(labels):
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(i)

        # Find minimum class size
        min_size = min(len(indices) for indices in label_groups.values())

        self.logger.info(f"Balancing classes to {min_size} samples each")

        # Sample equally from each class
        balanced_texts = []
        balanced_labels = []
        balanced_scenarios = []
        balanced_responses = []

        for label, indices in sorted(label_groups.items()):
            # Randomly sample min_size indices
            sampled_indices = np.random.choice(indices, size=min_size, replace=False)

            for idx in sampled_indices:
                balanced_texts.append(texts[idx])
                balanced_labels.append(labels[idx])
                balanced_scenarios.append(scenarios[idx])
                balanced_responses.append(responses[idx])

        # Shuffle
        combined = list(zip(balanced_texts, balanced_labels, balanced_scenarios, balanced_responses))
        np.random.shuffle(combined)
        balanced_texts, balanced_labels, balanced_scenarios, balanced_responses = zip(*combined)

        self.logger.info(f"Balanced dataset size: {len(balanced_texts)}")
        self._log_class_distribution(list(balanced_labels))

        return list(balanced_texts), list(balanced_labels), list(balanced_scenarios), list(balanced_responses)

    def _log_class_distribution(self, labels: List[int]):
        """Log the distribution of classes in the dataset"""
        class_counts = {}
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1

        total = len(labels)
        self.logger.info("Class distribution:")
        for class_id, count in sorted(class_counts.items()):
            if class_id in self.categories:
                category_name = self.categories[class_id]["name"]
                percentage = (count / total) * 100
                self.logger.info(f"  Class {class_id} ({category_name}): {count} samples ({percentage:.1f}%)")

    def create_train_val_test_split(
        self,
        texts: List[str],
        labels: List[int],
        scenarios: List[str],
        responses: List[str],
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1
    ) -> Tuple[List[str], List[int], List[str], List[str],
               List[str], List[int], List[str], List[str],
               List[str], List[int], List[str], List[str]]:
        """
        Split dataset into train, validation, and test sets

        Args:
            texts: List of texts
            labels: List of labels
            scenarios: List of scenarios
            responses: List of responses
            train_size: Proportion for training (default 0.8 = 80%)
            val_size: Proportion for validation (default 0.1 = 10%)
            test_size: Proportion for testing (default 0.1 = 10%)

        Returns:
            Tuple of (train_texts, train_labels, train_scenarios, train_responses,
                     val_texts, val_labels, val_scenarios, val_responses,
                     test_texts, test_labels, test_scenarios, test_responses)
        """
        # Combine texts, scenarios, and responses for splitting
        combined_data = list(zip(texts, scenarios, responses))

        # First split: train vs (val + test)
        train_data, temp_data, train_labels, temp_labels = train_test_split(
            combined_data, labels,
            test_size=(val_size + test_size),
            random_state=self.seed,
            stratify=labels
        )

        # Second split: val vs test
        val_data, test_data, val_labels, test_labels = train_test_split(
            temp_data, temp_labels,
            test_size=(test_size / (val_size + test_size)),
            random_state=self.seed,
            stratify=temp_labels
        )

        # Unzip data
        train_texts, train_scenarios, train_responses = zip(*train_data) if train_data else ([], [], [])
        val_texts, val_scenarios, val_responses = zip(*val_data) if val_data else ([], [], [])
        test_texts, test_scenarios, test_responses = zip(*test_data) if test_data else ([], [], [])

        self.logger.info(
            f"Dataset split - Train: {len(train_texts)}, "
            f"Val: {len(val_texts)}, Test: {len(test_texts)}"
        )

        return (
            list(train_texts), list(train_labels), list(train_scenarios), list(train_responses),
            list(val_texts), list(val_labels), list(val_scenarios), list(val_responses),
            list(test_texts), list(test_labels), list(test_scenarios), list(test_responses)
        )

    def save_dataset(
        self,
        texts: List[str],
        labels: List[int],
        scenarios: List[str],
        responses: List[str],
        filepath: str,
        include_metadata: bool = True
    ):
        """
        Save dataset to JSON file (compatible with RouterLLM format)

        Args:
            texts: List of texts
            labels: List of labels
            scenarios: List of scenarios
            responses: List of responses (correct answers)
            filepath: Path to save file
            include_metadata: Whether to include metadata
        """
        dataset = {
            "texts": texts,
            "labels": labels,
            "responses": responses,  # Store responses for future use
            "size": len(texts)
        }

        if include_metadata:
            dataset["metadata"] = {
                "source": self.dataset_name,
                "categories": self.categories,
                "seed": self.seed,
                "target_audience_mapping": self.target_audience_mapping,
                "class_distribution": self._get_class_distribution(labels),
                "scenarios": {
                    "unique_count": len(set(scenarios)),
                    "samples": scenarios[:10]  # First 10 for reference
                },
                "responses_info": {
                    "included": True,
                    "description": "Correct responses from the dataset (not used for training yet)"
                }
            }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Dataset saved to {filepath}")

    def load_saved_dataset(self, filepath: str) -> Tuple[List[str], List[int]]:
        """
        Load dataset from JSON file

        Args:
            filepath: Path to dataset file

        Returns:
            Tuple of (texts, labels)
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        texts = dataset["texts"]
        labels = dataset["labels"]

        self.logger.info(f"Dataset loaded from {filepath} - {len(texts)} samples")

        return texts, labels

    def _get_class_distribution(self, labels: List[int]) -> Dict[int, int]:
        """Get class distribution as dictionary"""
        class_counts = {}
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        return class_counts

    def generate_and_save_complete_dataset(
        self,
        output_dir: str = "./data",
        max_samples: Optional[int] = None,
        balance_classes: bool = True,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1
    ):
        """
        Load from HuggingFace and save complete dataset with train/val/test splits

        Args:
            output_dir: Directory to save datasets
            max_samples: Maximum samples to load (None = all)
            balance_classes: Whether to balance class distribution
            train_size: Training set proportion (default 0.8 = 80%)
            val_size: Validation set proportion (default 0.1 = 10%)
            test_size: Test set proportion (default 0.1 = 10%)
        """
        # Load dataset from HuggingFace
        texts, labels, scenarios, responses = self.load_huggingface_dataset(
            max_samples=max_samples,
            split="train"
        )

        # Balance classes if requested
        if balance_classes:
            texts, labels, scenarios, responses = self.balance_classes(texts, labels, scenarios, responses)

        # Split dataset
        (train_texts, train_labels, train_scenarios, train_responses,
         val_texts, val_labels, val_scenarios, val_responses,
         test_texts, test_labels, test_scenarios, test_responses) = self.create_train_val_test_split(
            texts, labels, scenarios, responses, train_size, val_size, test_size
        )

        # Save datasets
        os.makedirs(output_dir, exist_ok=True)

        self.save_dataset(
            train_texts, train_labels, train_scenarios, train_responses,
            f"{output_dir}/train_dataset.json"
        )
        self.save_dataset(
            val_texts, val_labels, val_scenarios, val_responses,
            f"{output_dir}/val_dataset.json"
        )
        self.save_dataset(
            test_texts, test_labels, test_scenarios, test_responses,
            f"{output_dir}/test_dataset.json"
        )

        # Save full dataset
        self.save_dataset(
            texts, labels, scenarios, responses,
            f"{output_dir}/full_dataset.json"
        )

        self.logger.info(f"Complete dataset saved to {output_dir}")

        # Print summary
        print(f"\n{'='*60}")
        print("DATASET GENERATION SUMMARY")
        print(f"{'='*60}")
        print(f"Source: {self.dataset_name}")
        print(f"Total samples: {len(texts)}")
        print(f"Train samples: {len(train_texts)} ({train_size*100:.0f}%)")
        print(f"Validation samples: {len(val_texts)} ({val_size*100:.0f}%)")
        print(f"Test samples: {len(test_texts)} ({test_size*100:.0f}%)")
        print(f"\nClass distribution (balanced: {balance_classes}):")
        for class_id, count in sorted(self._get_class_distribution(labels).items()):
            if class_id in self.categories:
                print(f"  Class {class_id} ({self.categories[class_id]['name']}): {count}")
        print(f"\nResponses included: Yes (for future use)")
        print(f"{'='*60}\n")

        return {
            "train": (train_texts, train_labels, train_scenarios, train_responses),
            "val": (val_texts, val_labels, val_scenarios, val_responses),
            "test": (test_texts, test_labels, test_scenarios, test_responses),
            "full": (texts, labels, scenarios, responses)
        }
