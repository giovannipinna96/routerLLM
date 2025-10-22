"""
LLM Manager for handling multiple Hugging Face models
"""

import gc
import torch
import psutil
import logging
from typing import Dict, List, Optional, Any, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
    pipeline
)
import yaml


class LLMManager:
    """
    Manager for multiple LLM models with GPU memory optimization
    """

    def __init__(
        self,
        config_path: str = "configs/default_config.yaml",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize LLM Manager

        Args:
            config_path: Path to configuration file
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config = self._load_config(config_path)

        # Model storage
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.pipelines: Dict[str, Any] = {}  # Store pipeline instances
        self.model_configs: Dict[str, Dict] = {}

        # Current loaded model
        self.current_model: Optional[str] = None

        # Device configuration
        self.device = self._setup_device()

        self.logger.info(f"LLMManager initialized with device: {self.device}")
        self._log_system_info()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            raise

    def _setup_device(self) -> str:
        """Setup compute device"""
        device_config = self.config.get("system", {}).get("device", "auto")

        if device_config == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                self.logger.info(f"CUDA available, using GPU: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                self.logger.info("CUDA not available, using CPU")
        else:
            device = device_config

        return device

    def _log_system_info(self):
        """Log system information"""
        # CPU info
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()

        self.logger.info(f"System Info - CPU cores: {cpu_count}, RAM: {memory.total / (1024**3):.1f}GB")

        # GPU info
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                self.logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}, Memory: {gpu_memory:.1f}GB")

    def _create_quantization_config(self, use_4bit: bool) -> Optional[BitsAndBytesConfig]:
        """Create quantization configuration for memory optimization"""
        if not use_4bit or self.device == "cpu":
            return None

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return [model["name"] for model in self.config["models"]["llms"]]

    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get model configuration information"""
        for model_config in self.config["models"]["llms"]:
            if model_config["name"] == model_name:
                return model_config
        return None

    def load_model(self, model_name: str) -> bool:
        """
        Load a specific model into memory

        Args:
            model_name: Name of the model to load

        Returns:
            True if successful, False otherwise
        """
        if model_name == self.current_model:
            self.logger.info(f"Model {model_name} already loaded")
            return True

        # Unload current model first
        if self.current_model:
            self.unload_current_model()

        # Find model configuration
        model_config = self.get_model_info(model_name)
        if not model_config:
            self.logger.error(f"Model configuration not found: {model_name}")
            return False

        try:
            self.logger.info(f"Loading model with pipeline: {model_name} ({model_config['model_id']})")

            # Create quantization config if needed
            quantization_config = self._create_quantization_config(model_config.get("use_4bit", False))

            # Pipeline kwargs
            pipeline_kwargs = {
                "model": model_config["model_id"],
                "task": "text-generation",
                "device_map": "auto" if self.device == "cuda" else None,
                "dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "trust_remote_code": True,
            }

            if quantization_config:
                pipeline_kwargs["model_kwargs"] = {"quantization_config": quantization_config}

            # Create pipeline
            self.logger.info(f"Creating pipeline for {model_name}")
            text_pipeline = pipeline(**pipeline_kwargs)

            # Configure tokenizer if needed
            if text_pipeline.tokenizer.pad_token is None:
                if text_pipeline.tokenizer.unk_token is not None:
                    text_pipeline.tokenizer.pad_token = text_pipeline.tokenizer.unk_token
                else:
                    text_pipeline.tokenizer.add_special_tokens({'pad_token': '<PAD>'})

            self.logger.info(f"Pipeline configured - pad_token: {text_pipeline.tokenizer.pad_token}, eos_token: {text_pipeline.tokenizer.eos_token}")

            # Store pipeline and configs
            self.pipelines[model_name] = text_pipeline
            self.models[model_name] = text_pipeline.model  # Keep for compatibility
            self.tokenizers[model_name] = text_pipeline.tokenizer  # Keep for compatibility
            self.model_configs[model_name] = model_config
            self.current_model = model_name

            # Log model info
            num_params = sum(p.numel() for p in text_pipeline.model.parameters())
            self.logger.info(f"Model {model_name} loaded with pipeline successfully. Parameters: {num_params:,}")

            self._log_memory_usage()
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            return False

    def unload_current_model(self):
        """Unload the currently loaded model to free memory"""
        if not self.current_model:
            return

        self.logger.info(f"Unloading model: {self.current_model}")

        # Delete model, tokenizer, and pipeline
        if self.current_model in self.models:
            del self.models[self.current_model]
        if self.current_model in self.tokenizers:
            del self.tokenizers[self.current_model]
        if self.current_model in self.pipelines:
            del self.pipelines[self.current_model]
        if self.current_model in self.model_configs:
            del self.model_configs[self.current_model]

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force garbage collection
        gc.collect()

        self.logger.info(f"Model {self.current_model} unloaded")
        self.current_model = None
        self._log_memory_usage()

    def _format_prompt_for_model(self, prompt: str, model_name: str) -> str:
        """
        Format prompt according to model requirements

        Args:
            prompt: Original prompt
            model_name: Name of the model

        Returns:
            Formatted prompt (currently no special formatting needed for Qwen3-Coder)
        """
        # For Qwen3-Coder-30B and other models, no special formatting is required
        # The model can handle standard prompts directly
        return prompt

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> Optional[str]:
        """
        Generate response using the currently loaded model pipeline

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling (lower = more focused)
            top_p: Top-p for nucleus sampling
            do_sample: Whether to use sampling

        Returns:
            Generated text or None if failed
        """
        if not self.current_model:
            self.logger.error("No model currently loaded")
            return None

        try:
            # Get pipeline for current model
            text_pipeline = self.pipelines[self.current_model]

            # Format prompt for model-specific requirements
            formatted_prompt = self._format_prompt_for_model(prompt, self.current_model)

            self.logger.info(f"Generating response with {self.current_model} pipeline")
            if formatted_prompt != prompt:
                self.logger.debug(f"Applied special formatting for {self.current_model}")

            # Generation parameters
            generation_params = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "return_full_text": False,  # Only return generated text, not input
                "repetition_penalty": 1.1,
            }

            # Add sampling parameters only if do_sample=True
            if do_sample:
                generation_params.update({
                    "temperature": temperature,
                    "top_p": top_p,
                })

            # Generate with pipeline - much simpler!
            outputs = text_pipeline(formatted_prompt, **generation_params)

            # Extract generated text
            if outputs and len(outputs) > 0:
                response = outputs[0].get("generated_text", "")
                self.logger.info(f"Response generated successfully. Length: {len(response)} chars")
                return response.strip()
            else:
                self.logger.warning("Pipeline returned empty output")
                return None

        except Exception as e:
            self.logger.error(f"Pipeline generation failed: {e}")
            return None

    def get_model_category(self, model_name: str) -> Optional[int]:
        """Get the category for a specific model"""
        model_config = self.get_model_info(model_name)
        return model_config.get("category") if model_config else None

    def _log_memory_usage(self):
        """Log current memory usage"""
        # RAM usage
        memory = psutil.virtual_memory()
        ram_used = (memory.total - memory.available) / (1024**3)

        self.logger.info(f"Memory usage - RAM: {ram_used:.1f}GB / {memory.total / (1024**3):.1f}GB")

        # GPU memory usage
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                cached = torch.cuda.memory_reserved(i) / (1024**3)
                total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                self.logger.info(f"GPU {i} memory - Allocated: {allocated:.1f}GB, Cached: {cached:.1f}GB, Total: {total:.1f}GB")

    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'current_model') and self.current_model:
            self.unload_current_model()