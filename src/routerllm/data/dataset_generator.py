"""
Synthetic dataset generator for router training
"""

import random
import json
import os
import logging
from typing import List, Dict, Tuple
import numpy as np
from sklearn.model_selection import train_test_split


class RouterDatasetGenerator:
    """
    Generator for synthetic router training datasets
    """

    def __init__(
        self,
        seed: int = 42,
        logger: logging.Logger = None
    ):
        """
        Initialize dataset generator

        Args:
            seed: Random seed for reproducibility
            logger: Logger instance
        """
        self.seed = seed
        self.logger = logger or logging.getLogger(__name__)

        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)

        # Define categories and their characteristics
        self.categories = {
            0: {  # Code generation - CodeLlama
                "name": "code_generation",
                "model": "code_llama_7b",
                "keywords": [
                    "write a function", "implement", "code", "programming", "algorithm",
                    "debug", "fix the bug", "create a class", "write a script", "python",
                    "javascript", "java", "c++", "sql", "api", "function", "method",
                    "loop", "condition", "variable", "array", "list", "dictionary"
                ],
                "templates": [
                    "Write a {language} function that {task}",
                    "Implement an algorithm to {task}",
                    "Create a {language} script that {task}",
                    "How do I {task} in {language}?",
                    "Debug this {language} code: {code_snippet}",
                    "Write a function to {task} using {approach}",
                    "Implement a {data_structure} in {language}",
                    "Create a {language} program that {task}",
                    "Fix the bug in this code: {code_snippet}",
                    "Write unit tests for {function_name}"
                ]
            },
            1: {  # Text generation - Phi-3
                "name": "text_generation",
                "model": "phi3_mini",
                "keywords": [
                    "write a story", "creative writing", "article", "essay", "blog post",
                    "content", "narrative", "fiction", "non-fiction", "poem", "letter",
                    "email", "report", "summary", "description", "review", "opinion"
                ],
                "templates": [
                    "Write a {text_type} about {topic}",
                    "Create a {length} {text_type} on {topic}",
                    "Generate content for a {text_type} about {topic}",
                    "Write a creative {text_type} involving {elements}",
                    "Compose a {style} {text_type} about {topic}",
                    "Create a {text_type} that explains {concept}",
                    "Write a persuasive {text_type} about {topic}",
                    "Generate a {text_type} for {audience} about {topic}",
                    "Create engaging content about {topic}",
                    "Write a detailed {text_type} covering {aspects}"
                ]
            },
            2: {  # General purpose - Mistral
                "name": "general_purpose",
                "model": "mistral_7b",
                "keywords": [
                    "explain", "analyze", "compare", "discuss", "what is", "how does",
                    "why", "when", "where", "definition", "difference", "advantages",
                    "disadvantages", "pros and cons", "impact", "effect", "cause",
                    "solution", "recommendation", "advice", "guidance", "help"
                ],
                "templates": [
                    "Explain {concept} in simple terms",
                    "What are the main differences between {item1} and {item2}?",
                    "Analyze the impact of {factor} on {domain}",
                    "How does {process} work?",
                    "What are the advantages and disadvantages of {topic}?",
                    "Compare {option1} and {option2} for {use_case}",
                    "Provide a comprehensive overview of {topic}",
                    "What factors should I consider when {decision}?",
                    "Explain the relationship between {concept1} and {concept2}",
                    "How can I improve {skill} in {domain}?"
                ]
            },
            3: {  # Lightweight tasks - TinyLlama
                "name": "lightweight_tasks",
                "model": "tiny_llama",
                "keywords": [
                    "translate", "convert", "format", "list", "simple", "quick",
                    "basic", "elementary", "straightforward", "easy", "simple task",
                    "brief", "short", "minimal", "fast", "simple question"
                ],
                "templates": [
                    "List the main {items} of {topic}",
                    "What is {simple_concept}?",
                    "Translate '{text}' to {language}",
                    "Convert {value} from {unit1} to {unit2}",
                    "Give me a brief overview of {topic}",
                    "What are the key points about {topic}?",
                    "Provide a simple explanation of {concept}",
                    "List {number} {items} for {purpose}",
                    "What is the definition of {term}?",
                    "Give me quick facts about {topic}"
                ]
            }
        }

        # Template variables
        self.variables = {
            "language": ["Python", "JavaScript", "Java", "C++", "Go", "Rust", "SQL", "HTML", "CSS"],
            "task": [
                "calculate the factorial of a number", "sort an array", "find the maximum value",
                "reverse a string", "check if a number is prime", "merge two lists",
                "find duplicates in a list", "calculate fibonacci sequence",
                "implement binary search", "parse JSON data"
            ],
            "approach": [
                "recursion", "iteration", "dynamic programming", "greedy algorithm",
                "divide and conquer", "brute force", "optimization"
            ],
            "data_structure": [
                "linked list", "binary tree", "hash table", "queue", "stack",
                "graph", "heap", "trie"
            ],
            "code_snippet": [
                "for i in range(len(arr)): print(arr[i])",
                "def calculate(x, y): return x + y",
                "if condition: result = True else: result = False"
            ],
            "function_name": [
                "calculate_average", "find_minimum", "sort_data", "process_input",
                "validate_email", "format_output"
            ],
            "text_type": [
                "story", "article", "essay", "blog post", "report", "letter", "email",
                "poem", "review", "summary", "description", "tutorial"
            ],
            "topic": [
                "artificial intelligence", "climate change", "space exploration",
                "renewable energy", "cybersecurity", "blockchain", "machine learning",
                "quantum computing", "biotechnology", "sustainable development"
            ],
            "length": ["short", "medium", "long", "brief", "detailed", "comprehensive"],
            "style": ["formal", "informal", "academic", "conversational", "technical"],
            "audience": ["beginners", "experts", "students", "professionals", "general public"],
            "concept": [
                "machine learning", "blockchain", "quantum physics", "evolution",
                "photosynthesis", "democracy", "capitalism", "neural networks"
            ],
            "item1": ["Python", "Java", "iOS", "Windows", "MySQL", "PostgreSQL"],
            "item2": ["JavaScript", "C++", "Android", "Linux", "MongoDB", "Redis"],
            "domain": [
                "technology", "healthcare", "education", "finance", "environment",
                "business", "society", "economy"
            ],
            "process": [
                "photosynthesis", "democracy", "machine learning", "blockchain",
                "cryptocurrency mining", "neural network training"
            ],
            "factor": [
                "artificial intelligence", "climate change", "globalization",
                "digitalization", "automation", "social media"
            ],
            "option1": ["cloud computing", "traditional hosting", "iOS", "SQL"],
            "option2": ["on-premise servers", "serverless architecture", "Android", "NoSQL"],
            "use_case": [
                "small businesses", "enterprise applications", "mobile development",
                "data analysis", "web development"
            ],
            "decision": [
                "choosing a programming language", "selecting a database",
                "picking a cloud provider", "deciding on architecture"
            ],
            "concept1": ["artificial intelligence", "machine learning", "data science"],
            "concept2": ["automation", "decision making", "business intelligence"],
            "skill": ["programming", "data analysis", "communication", "problem solving"],
            "items": ["features", "benefits", "components", "principles", "elements"],
            "simple_concept": [
                "artificial intelligence", "blockchain", "cloud computing",
                "machine learning", "cryptocurrency"
            ],
            "text": ["Hello, how are you?", "Good morning", "Thank you very much"],
            "language": ["Spanish", "French", "German", "Italian", "Portuguese"],
            "value": ["100", "25.4", "1000", "5.5"],
            "unit1": ["meters", "feet", "kilograms", "pounds", "celsius"],
            "unit2": ["feet", "meters", "pounds", "kilograms", "fahrenheit"],
            "number": ["5", "10", "3", "7"],
            "purpose": [
                "productivity", "learning", "health", "business", "entertainment"
            ],
            "term": [
                "API", "algorithm", "database", "framework", "library",
                "protocol", "interface", "methodology"
            ]
        }

        self.logger.info(f"DatasetGenerator initialized with {len(self.categories)} categories")

    def generate_sample(self, category_id: int) -> Tuple[str, int]:
        """
        Generate a single sample for a specific category

        Args:
            category_id: Category ID (0-3)

        Returns:
            Tuple of (text, label)
        """
        if category_id not in self.categories:
            raise ValueError(f"Invalid category_id: {category_id}")

        category = self.categories[category_id]

        # Choose a random template
        template = random.choice(category["templates"])

        # Fill template with random variables
        text = template
        for var_name, var_options in self.variables.items():
            placeholder = "{" + var_name + "}"
            if placeholder in text:
                text = text.replace(placeholder, random.choice(var_options))

        # Add some natural variations
        text = self._add_variations(text, category)

        return text, category_id

    def _add_variations(self, text: str, category: Dict) -> str:
        """Add natural variations to the text"""
        # Sometimes add politeness
        if random.random() < 0.3:
            politeness = ["Please", "Could you", "Can you", "Would you mind"]
            text = f"{random.choice(politeness)} {text.lower()}"

        # Sometimes add context
        if random.random() < 0.2:
            context = [
                "I need help with this:",
                "I'm working on a project and",
                "For my assignment,",
                "I'm trying to understand"
            ]
            text = f"{random.choice(context)} {text.lower()}"

        # Sometimes add urgency/importance
        if random.random() < 0.1:
            urgency = [
                "It's urgent:",
                "This is important:",
                "I need this quickly:",
                "Time-sensitive:"
            ]
            text = f"{random.choice(urgency)} {text.lower()}"

        return text

    def generate_dataset(
        self,
        total_samples: int = 1000,
        balance_classes: bool = True
    ) -> Tuple[List[str], List[int]]:
        """
        Generate a complete dataset

        Args:
            total_samples: Total number of samples to generate
            balance_classes: Whether to balance classes

        Returns:
            Tuple of (texts, labels)
        """
        texts = []
        labels = []

        if balance_classes:
            samples_per_class = total_samples // len(self.categories)
            remaining_samples = total_samples % len(self.categories)

            for category_id in range(len(self.categories)):
                # Generate base samples for this class
                class_samples = samples_per_class

                # Distribute remaining samples
                if category_id < remaining_samples:
                    class_samples += 1

                for _ in range(class_samples):
                    text, label = self.generate_sample(category_id)
                    texts.append(text)
                    labels.append(label)
        else:
            # Random distribution
            for _ in range(total_samples):
                category_id = random.randint(0, len(self.categories) - 1)
                text, label = self.generate_sample(category_id)
                texts.append(text)
                labels.append(label)

        # Shuffle the dataset
        combined = list(zip(texts, labels))
        random.shuffle(combined)
        texts, labels = zip(*combined)

        self.logger.info(f"Generated dataset with {len(texts)} samples")
        self._log_class_distribution(labels)

        return list(texts), list(labels)

    def _log_class_distribution(self, labels: List[int]):
        """Log the distribution of classes in the dataset"""
        class_counts = {}
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1

        total = len(labels)
        self.logger.info("Class distribution:")
        for class_id, count in sorted(class_counts.items()):
            category_name = self.categories[class_id]["name"]
            percentage = (count / total) * 100
            self.logger.info(f"  Class {class_id} ({category_name}): {count} samples ({percentage:.1f}%)")

    def create_train_val_test_split(
        self,
        texts: List[str],
        labels: List[int],
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15
    ) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
        """
        Split dataset into train, validation, and test sets

        Args:
            texts: List of texts
            labels: List of labels
            train_size: Proportion for training
            val_size: Proportion for validation
            test_size: Proportion for testing

        Returns:
            Tuple of (train_texts, train_labels, val_texts, val_labels, test_texts, test_labels)
        """
        # First split: train vs (val + test)
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels,
            test_size=(val_size + test_size),
            random_state=self.seed,
            stratify=labels
        )

        # Second split: val vs test
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels,
            test_size=(test_size / (val_size + test_size)),
            random_state=self.seed,
            stratify=temp_labels
        )

        self.logger.info(f"Dataset split - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

        return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

    def save_dataset(
        self,
        texts: List[str],
        labels: List[int],
        filepath: str,
        include_metadata: bool = True
    ):
        """
        Save dataset to JSON file

        Args:
            texts: List of texts
            labels: List of labels
            filepath: Path to save file
            include_metadata: Whether to include metadata
        """
        dataset = {
            "texts": texts,
            "labels": labels,
            "size": len(texts)
        }

        if include_metadata:
            dataset["metadata"] = {
                "categories": self.categories,
                "seed": self.seed,
                "class_distribution": self._get_class_distribution(labels)
            }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Dataset saved to {filepath}")

    def load_dataset(self, filepath: str) -> Tuple[List[str], List[int]]:
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
        total_samples: int = 1200,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15
    ):
        """
        Generate and save a complete dataset with train/val/test splits

        Args:
            output_dir: Directory to save datasets
            total_samples: Total number of samples
            train_size: Training set proportion
            val_size: Validation set proportion
            test_size: Test set proportion
        """
        # Generate full dataset
        texts, labels = self.generate_dataset(total_samples, balance_classes=True)

        # Split dataset
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
            self.create_train_val_test_split(texts, labels, train_size, val_size, test_size)

        # Save datasets
        os.makedirs(output_dir, exist_ok=True)

        self.save_dataset(train_texts, train_labels, f"{output_dir}/train_dataset.json")
        self.save_dataset(val_texts, val_labels, f"{output_dir}/val_dataset.json")
        self.save_dataset(test_texts, test_labels, f"{output_dir}/test_dataset.json")

        # Save full dataset
        self.save_dataset(texts, labels, f"{output_dir}/full_dataset.json")

        self.logger.info(f"Complete dataset saved to {output_dir}")

        return {
            "train": (train_texts, train_labels),
            "val": (val_texts, val_labels),
            "test": (test_texts, test_labels),
            "full": (texts, labels)
        }