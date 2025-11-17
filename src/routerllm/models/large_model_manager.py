"""
Large Model Manager for handling 100B+ parameter models
Supports multi-GPU deployment and advanced memory optimization
"""

import gc
import torch
import psutil
import logging
from typing import Dict, List, Optional, Any, Union
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
    pipeline
)
import accelerate
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map


class LargeModelManager:
    """
    Manager for very large language models (100B+ parameters)
    Implements advanced memory management and multi-GPU support
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Large Model Manager
        
        Args:
            model_config: Model configuration dictionary
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.model_config = model_config
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # Model metadata
        self.model_name = model_config.get("name", "unknown")
        self.model_id = model_config.get("model_id", "")
        self.use_4bit = model_config.get("use_4bit", True)  # Required for 100B+ models
        self.use_flash_attention = model_config.get("use_flash_attention", True)
        
        # Multi-GPU configuration
        self.device_map = model_config.get("device_map", "auto")
        self.max_memory = model_config.get("max_memory", None)
        
        # Setup devices
        self.num_gpus = torch.cuda.device_count()
        self.logger.info(f"LargeModelManager initialized for {self.model_name}")
        self.logger.info(f"Available GPUs: {self.num_gpus}")
        
        if self.num_gpus > 0:
            for i in range(self.num_gpus):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                self.logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}, Memory: {gpu_memory:.1f}GB")
                
    def _create_quantization_config(self) -> BitsAndBytesConfig:
        """
        Create aggressive quantization config for 100B+ models
        
        Returns:
            BitsAndBytesConfig for 4-bit quantization
        """
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    
    def _validate_gpu_availability(self) -> bool:
        """
        Validate GPU availability and memory for large model loading

        Returns:
            True if setup is adequate, False otherwise
        """
        if self.num_gpus == 0:
            self.logger.error("❌ No GPUs detected. Large models (100B+) require GPU acceleration.")
            return False

        # Calculate total available GPU memory
        total_gpu_memory_gb = 0
        for i in range(self.num_gpus):
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            total_gpu_memory_gb += gpu_mem

        self.logger.info(f"Total GPU memory available: {total_gpu_memory_gb:.1f}GB across {self.num_gpus} GPU(s)")

        # Estimate required memory for 405B model
        # With 4-bit quantization: ~200-250GB minimum
        required_memory_gb = 200 if self.use_4bit else 800

        if total_gpu_memory_gb < required_memory_gb:
            self.logger.warning(f"⚠️  Limited GPU memory: {total_gpu_memory_gb:.1f}GB available, {required_memory_gb}GB recommended")
            self.logger.warning("Will attempt loading with CPU offload and aggressive memory management")
            if not self.use_4bit:
                self.logger.error("❌ 4-bit quantization REQUIRED for this hardware configuration")
                return False
        else:
            self.logger.info(f"✓ GPU memory adequate: {total_gpu_memory_gb:.1f}GB available")

        return True

    def _setup_device_map(self) -> Dict:
        """
        Setup device map for multi-GPU deployment with CPU fallback

        Returns:
            Device map for model sharding
        """
        if self.max_memory:
            # Use provided max_memory configuration
            self.logger.info("Using provided max_memory configuration")
            return self.max_memory

        # Auto-calculate based on available GPUs
        if self.num_gpus == 0:
            self.logger.warning("No GPUs available, forcing CPU-only mode")
            return {"": "cpu"}

        # Distribute model across GPUs with aggressive memory usage
        max_memory_per_gpu = {}
        for i in range(self.num_gpus):
            # For 2-3 GPU setup, use 95% of available memory (more aggressive)
            available_memory = torch.cuda.get_device_properties(i).total_memory
            # Use 95% for 2-3 GPU setups, 90% for larger setups
            usage_ratio = 0.95 if self.num_gpus <= 3 else 0.90
            reserved_memory = int(available_memory * usage_ratio)
            max_memory_per_gpu[i] = f"{reserved_memory}B"
            self.logger.info(f"GPU {i}: Allocating {reserved_memory / (1024**3):.1f}GB")

        # Add CPU memory as fallback (important for 2-3 GPU setups!)
        cpu_memory = psutil.virtual_memory().total
        # Use up to 70% of RAM for CPU offload
        cpu_offload = int(cpu_memory * 0.70)
        max_memory_per_gpu["cpu"] = f"{cpu_offload}B"
        self.logger.info(f"CPU: Allocating {cpu_offload / (1024**3):.1f}GB for offload")

        self.logger.info(f"Device memory map configured: {len(max_memory_per_gpu)} devices")
        return max_memory_per_gpu
        
    def _get_no_split_module_classes(self) -> List[str]:
        """
        Get the list of module classes that should not be split across devices

        Returns:
            List of module class names
        """
        # Map model families to their decoder layer class names
        model_id_lower = self.model_id.lower()

        if "llama" in model_id_lower:
            return ["LlamaDecoderLayer"]
        elif "falcon" in model_id_lower:
            return ["FalconDecoderLayer"]
        elif "bloom" in model_id_lower:
            return ["BloomBlock"]
        elif "gpt" in model_id_lower or "starcoder" in model_id_lower:
            return ["GPTBigCodeBlock", "GPTNeoXLayer"]
        elif "mistral" in model_id_lower:
            return ["MistralDecoderLayer"]
        else:
            # Generic fallback - try common names
            self.logger.warning(f"Unknown model family for {self.model_id}, using generic no_split classes")
            return ["DecoderLayer", "TransformerBlock"]

    def load_model(self) -> bool:
        """
        Load the large model with advanced Accelerate-based optimization

        Uses explicit device mapping with infer_auto_device_map for better control
        over model sharding across GPUs and CPU offload.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("="*80)
            self.logger.info(f"🚀 Loading Large Model: {self.model_id}")
            self.logger.info("="*80)

            # Step 0: Validate GPU availability
            if not self._validate_gpu_availability():
                return False

            # Step 1: Load tokenizer
            self.logger.info("📝 Step 1/5: Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                use_fast=True
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.logger.info("✓ Tokenizer loaded successfully")

            # Step 2: Setup quantization
            self.logger.info(f"⚙️  Step 2/5: Configuring {'4-bit' if self.use_4bit else 'FP16'} quantization...")
            quantization_config = self._create_quantization_config() if self.use_4bit else None
            if quantization_config:
                self.logger.info("✓ 4-bit quantization enabled (NF4 with double quantization)")

            # Step 3: Setup max memory for device mapping
            self.logger.info("💾 Step 3/5: Configuring device memory allocation...")
            max_memory = self._setup_device_map() if self.device_map == "auto" else self.device_map

            # Step 4: Initialize model with empty weights to infer device map
            self.logger.info("🗺️  Step 4/5: Computing optimal device map with Accelerate...")
            self.logger.info("This may take 2-5 minutes for 405B models...")

            # Get no-split module classes for this model architecture
            no_split_modules = self._get_no_split_module_classes()
            self.logger.info(f"No-split modules: {no_split_modules}")

            # Configure offload directory
            offload_folder = self.model_config.get("offload_folder", "./offload")
            import os
            os.makedirs(offload_folder, exist_ok=True)
            self.logger.info(f"Offload folder: {offload_folder}")

            # Load model with Accelerate-based device mapping
            model_kwargs = {
                "pretrained_model_name_or_path": self.model_id,
                "torch_dtype": torch.float16,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "max_memory": max_memory,
                "offload_folder": offload_folder,
                "offload_state_dict": True,  # Offload state dict to disk during loading
            }

            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"  # Let quantization handle device placement
            else:
                # Use explicit device map computation for non-quantized models
                # First, initialize with empty weights to compute device map
                with init_empty_weights():
                    empty_model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                    )

                # Compute optimal device map
                device_map = infer_auto_device_map(
                    empty_model,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_modules,
                    dtype=torch.float16
                )
                model_kwargs["device_map"] = device_map

                # Log device map distribution
                self.logger.info("Device map computed:")
                device_counts = {}
                for layer, device in device_map.items():
                    device_counts[str(device)] = device_counts.get(str(device), 0) + 1
                for device, count in device_counts.items():
                    self.logger.info(f"  {device}: {count} layers")

            if self.use_flash_attention:
                try:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    self.logger.info("✓ Flash Attention 2 enabled")
                except Exception as e:
                    self.logger.warning(f"Flash Attention 2 not available: {e}")

            # Step 5: Load the actual model
            self.logger.info("📥 Step 5/5: Loading model weights...")
            self.logger.info("This will take 10-20 minutes for 405B models with 4-bit quantization")
            self.logger.info("Please be patient... ☕")

            self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

            # Create pipeline for easier inference
            self.logger.info("Creating text-generation pipeline...")
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )

            # Log successful loading
            self.logger.info("="*80)
            self.logger.info("✅ MODEL LOADED SUCCESSFULLY!")
            num_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Parameters: {num_params/1e9:.1f}B")
            self.logger.info("="*80)

            # Log memory usage
            self._log_memory_usage()

            return True

        except torch.cuda.OutOfMemoryError as e:
            self.logger.error("="*80)
            self.logger.error("❌ OUT OF MEMORY ERROR")
            self.logger.error("="*80)
            self.logger.error(f"Failed to load {self.model_id}: {e}")
            self.logger.error("\n💡 SOLUTIONS:")
            self.logger.error("1. ✓ Ensure 4-bit quantization is enabled (use_4bit: true)")
            self.logger.error("2. ✓ Reduce number of GPUs being used (free up memory)")
            self.logger.error("3. ✓ Use CPU offload (increase 'cpu' in max_memory)")
            self.logger.error("4. ✓ Use a smaller model (e.g., 70B or 180B instead of 405B)")
            self.logger.error("5. ✓ Close other GPU-using processes")
            self._log_memory_usage()
            return False

        except Exception as e:
            self.logger.error("="*80)
            self.logger.error("❌ MODEL LOADING FAILED")
            self.logger.error("="*80)
            self.logger.error(f"Error loading {self.model_id}: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            import traceback
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            self.logger.error("\n💡 TROUBLESHOOTING:")
            self.logger.error("1. Check internet connection (model downloads from HuggingFace)")
            self.logger.error("2. Verify model_id is correct in config")
            self.logger.error("3. Ensure HuggingFace token is set if model requires authentication")
            self.logger.error("4. Check CUDA/GPU drivers are properly installed")
            return False
            
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> Optional[str]:
        """
        Generate response using the large model
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p for nucleus sampling
            do_sample: Whether to use sampling
            
        Returns:
            Generated text or None if failed
        """
        if not self.model or not self.pipeline:
            self.logger.error("Model not loaded")
            return None
            
        try:
            self.logger.info(f"Generating with {self.model_name} (prompt length: {len(prompt)} chars)")
            
            # Generation parameters optimized for code generation
            generation_params = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "return_full_text": False,
                "use_cache": False,  # Disable cache to avoid DynamicCache.seen_tokens bug
            }
            
            # Generate
            with torch.no_grad():
                outputs = self.pipeline(prompt, **generation_params)
                
            if outputs and len(outputs) > 0:
                response = outputs[0]["generated_text"]
                self.logger.info(f"Generated {len(response)} chars")
                return response
            else:
                self.logger.warning("Empty response from model")
                return None
                
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return None
            
    def unload_model(self):
        """Unload the model to free memory"""
        if self.model:
            self.logger.info(f"Unloading model {self.model_name}")
            
            # Delete model components
            del self.model
            del self.tokenizer
            del self.pipeline
            
            self.model = None
            self.tokenizer = None  
            self.pipeline = None
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            # Force garbage collection
            gc.collect()
            
            self.logger.info("Model unloaded successfully")
            self._log_memory_usage()
            
    def _log_memory_usage(self):
        """Log current memory usage"""
        # CPU memory
        memory = psutil.virtual_memory()
        ram_used = (memory.total - memory.available) / (1024**3)
        self.logger.info(f"CPU Memory: {ram_used:.1f}GB / {memory.total / (1024**3):.1f}GB")
        
        # GPU memory
        if torch.cuda.is_available():
            for i in range(self.num_gpus):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                self.logger.info(f"GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved / {total:.1f}GB total")
                
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = {
            "name": self.model_name,
            "model_id": self.model_id,
            "loaded": self.model is not None,
            "use_4bit": self.use_4bit,
            "use_flash_attention": self.use_flash_attention,
            "num_gpus": self.num_gpus,
        }
        
        if self.model:
            num_params = sum(p.numel() for p in self.model.parameters())
            info["parameters"] = num_params
            info["parameters_billions"] = num_params / 1e9
            
        return info


class DirectLargeModelSystem:
    """
    System for running a single large model (100B+) for comparison
    """
    
    def __init__(
        self,
        config_path: str,
        enable_carbon_tracking: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Direct Large Model System
        
        Args:
            config_path: Path to configuration file
            enable_carbon_tracking: Whether to track carbon emissions
            logger: Logger instance
        """
        self.config_path = config_path
        self.logger = logger or logging.getLogger(__name__)
        
        # Load configuration
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Get large model configuration
        self.model_config = self.config['models']['large_llm']
        
        # Initialize components
        self.model_manager = None
        self.carbon_tracker = None
        
        # Setup carbon tracking
        if enable_carbon_tracking:
            from ..utils.carbon_tracker import CarbonTracker
            self.carbon_tracker = CarbonTracker(
                project_name="large_llm_direct",
                output_dir="./logs/carbon"
            )
            
        self.logger.info(f"DirectLargeModelSystem initialized for {self.model_config['name']}")
        
    def initialize(self):
        """Initialize the system"""
        self.logger.info("Initializing Direct Large Model System...")
        
        # Create model manager
        self.model_manager = LargeModelManager(
            self.model_config,
            logger=self.logger
        )
        
        # Load the model
        success = self.model_manager.load_model()
        if not success:
            raise RuntimeError("Failed to load large model")
            
        self.logger.info("Direct Large Model System initialized successfully")
        
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> Dict[str, Any]:
        """
        Generate response using the large model
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p for nucleus sampling
            do_sample: Whether to use sampling
            
        Returns:
            Response dictionary with generated text and metadata
        """
        import time
        import uuid
        
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        self.logger.info(f"Processing request {request_id} with large model")
        
        # Track carbon emissions
        with self.carbon_tracker.track_emissions("large_model_inference") if self.carbon_tracker else self._dummy_context():
            response = self.model_manager.generate_response(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample
            )
            
        total_time = time.time() - start_time
        
        return {
            "request_id": request_id,
            "response": response,
            "model_used": self.model_config['name'],
            "model_parameters": f"{self.model_config.get('parameters', '100B+')}",
            "total_time": total_time,
            "status": "success" if response else "error"
        }
        
    def _dummy_context(self):
        """Dummy context manager"""
        class DummyContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return DummyContext()
        
    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up Direct Large Model System...")
        
        if self.model_manager:
            self.model_manager.unload_model()
            
        if self.carbon_tracker:
            self.carbon_tracker.log_summary()
            
        self.logger.info("Cleanup completed")
        
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            "model_info": self.model_manager.get_model_info() if self.model_manager else {},
        }
        
        if self.carbon_tracker:
            stats["carbon_footprint"] = {
                "total_emissions_kg": self.carbon_tracker.get_total_emissions(),
                "emissions_breakdown": self.carbon_tracker.get_emissions_breakdown()
            }
            
        return stats
