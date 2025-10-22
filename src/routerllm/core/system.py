"""
Main RouterLLM System Integration
"""

import time
import uuid
import psutil
import gc
import torch
import logging
from typing import Dict, Any, Optional, Union, Tuple
import yaml

from ..models.llm_manager import LLMManager
from ..models.router import BERTRouter, DummyRouter, GrahamComplexityRouter, BaseRouter
from ..utils.carbon_tracker import CarbonTracker
from ..utils.logger import setup_logger, log_performance_metrics, log_request_response


class RouterLLMSystem:
    """
    Main system that integrates router and LLM manager for intelligent routing
    """

    def __init__(
        self,
        config_path: str = "configs/default_config.yaml",
        router_type: str = "dummy",  # "dummy", "bert", "nvidia_complexity", "graham_complexity"
        router_model_path: Optional[str] = None,
        enable_carbon_tracking: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize RouterLLM System

        Args:
            config_path: Path to configuration file
            router_type: Type of router to use ("dummy", "bert", "nvidia_complexity", "graham_complexity")
            router_model_path: Path to trained router model (for BERT router)
            enable_carbon_tracking: Whether to enable carbon footprint tracking
            logger: Logger instance
        """
        self.config_path = config_path
        self.router_type = router_type
        self.router_model_path = router_model_path
        self.enable_carbon_tracking = enable_carbon_tracking

        # Load configuration
        self.config = self._load_config()

        # Setup logging
        if logger is None:
            log_config = self.config.get("logging", {})
            self.logger = setup_logger(
                name="routerllm_system",
                log_dir=log_config.get("log_dir", "./logs"),
                log_file=log_config.get("log_file", "routerllm.log"),
                level=log_config.get("level", "INFO")
            )
        else:
            self.logger = logger

        # Initialize carbon tracker
        self.carbon_tracker = None
        if self.enable_carbon_tracking:
            carbon_config = self.config.get("carbon_tracking", {})
            self.carbon_tracker = CarbonTracker(
                project_name=carbon_config.get("project_name", "routerllm"),
                output_dir=carbon_config.get("output_dir", "./logs/carbon"),
                country_iso_code=carbon_config.get("country_iso_code", "USA"),
                offline_mode=not carbon_config.get("enabled", True),
                logger=self.logger
            )

        # Initialize components
        self.llm_manager = None
        self.router = None

        # System state
        self.is_initialized = False
        self.request_count = 0
        self.total_inference_time = 0.0
        self.total_loading_time = 0.0

        self.logger.info(f"RouterLLMSystem initialized - Router: {router_type}, Carbon tracking: {enable_carbon_tracking}")

    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Failed to load config from {self.config_path}: {e}")
            raise

    def initialize(self):
        """Initialize all system components"""
        if self.is_initialized:
            self.logger.warning("System already initialized")
            return

        self.logger.info("Initializing RouterLLM System...")

        try:
            # Initialize LLM Manager
            self.logger.info("Initializing LLM Manager...")
            self.llm_manager = LLMManager(
                config_path=self.config_path,
                logger=self.logger
            )

            # Initialize Router
            self.logger.info(f"Initializing {self.router_type} router...")
            if self.router_type == "dummy":
                self.router = DummyRouter(
                    num_classes=self.config["models"]["router"]["num_classes"],
                    seed=42,
                    logger=self.logger
                )
            elif self.router_type == "bert":
                router_config = self.config["models"]["router"]
                self.router = BERTRouter(
                    model_name=router_config["model_name"],
                    num_classes=router_config["num_classes"],
                    max_length=router_config["max_length"],
                    logger=self.logger
                )

                # Load trained model if path provided
                if self.router_model_path:
                    self.logger.info(f"Loading router model from {self.router_model_path}")
                    self.router.load_model(self.router_model_path)
                else:
                    self.logger.warning("No router model path provided for BERT router")
            elif self.router_type == "graham_complexity":
                complexity_config = self.config["models"].get("complexity_routers", {}).get("graham", {})
                model_name = complexity_config.get("model_name", "grahamaco/question-complexity-classifier")
                self.router = GrahamComplexityRouter(
                    model_name=model_name,
                    logger=self.logger
                )
            else:
                raise ValueError(f"Unknown router type: {self.router_type}")

            self.logger.info(f"{self.router_type} router initialized successfully")

            self.is_initialized = True
            self.logger.info("RouterLLM System initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            raise

    def predict_and_generate(
        self,
        input_text: str,
        max_length: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> Dict[str, Any]:
        """
        Main method: predict best LLM and generate response

        Args:
            input_text: Input text/prompt
            max_length: Maximum number of new tokens to generate
            temperature: Temperature for sampling (lower = more deterministic)
            top_p: Top-p for nucleus sampling
            do_sample: Whether to use sampling

        Returns:
            Dictionary containing response and metadata
        """
        if not self.is_initialized:
            self.initialize()

        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        self.logger.info(f"Processing request {request_id}")

        try:
            # Step 1: Router prediction
            with self.carbon_tracker.track_emissions("router_inference") if self.carbon_tracker else self._dummy_context():
                router_start = time.time()
                predicted_class, confidence = self.router.predict(input_text)
                router_time = time.time() - router_start

                predicted_model = self.router.get_model_name_from_class(predicted_class)

            self.logger.info(f"Router prediction: {predicted_model} (class {predicted_class}, confidence {confidence:.4f})")

            # Step 2: Load the selected LLM
            with self.carbon_tracker.track_emissions("model_loading") if self.carbon_tracker else self._dummy_context():
                loading_start = time.time()
                load_success = self.llm_manager.load_model(predicted_model)
                loading_time = time.time() - loading_start

                if not load_success:
                    raise RuntimeError(f"Failed to load model: {predicted_model}")

            self.logger.info(f"Model loaded successfully: {predicted_model}")

            # Step 3: Generate response
            with self.carbon_tracker.track_emissions("llm_inference") if self.carbon_tracker else self._dummy_context():
                generation_start = time.time()
                response = self.llm_manager.generate_response(
                    prompt=input_text,
                    max_new_tokens=max_length,  # Renamed for clarity
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample
                )
                generation_time = time.time() - generation_start

                if response is None:
                    raise RuntimeError("Failed to generate response")

            total_time = time.time() - start_time

            # Log performance metrics
            self._log_performance_metrics(
                request_id=request_id,
                router_time=router_time,
                loading_time=loading_time,
                generation_time=generation_time,
                total_time=total_time
            )

            # Log request/response
            log_request_response(
                self.logger,
                request_id=request_id,
                input_text=input_text,
                predicted_model=predicted_model,
                response=response,
                confidence=confidence
            )

            # Update system statistics
            self.request_count += 1
            self.total_inference_time += generation_time
            self.total_loading_time += loading_time

            # Prepare result
            result = {
                "request_id": request_id,
                "input_text": input_text,
                "predicted_model": predicted_model,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "response": response,
                "timing": {
                    "router_time": router_time,
                    "loading_time": loading_time,
                    "generation_time": generation_time,
                    "total_time": total_time
                },
                "status": "success"
            }

            self.logger.info(f"Request {request_id} completed successfully in {total_time:.4f}s")
            return result

        except Exception as e:
            error_time = time.time() - start_time
            self.logger.error(f"Request {request_id} failed after {error_time:.4f}s: {e}")

            return {
                "request_id": request_id,
                "input_text": input_text,
                "error": str(e),
                "timing": {
                    "total_time": error_time
                },
                "status": "error"
            }

    def _dummy_context(self):
        """Dummy context manager when carbon tracking is disabled"""
        class DummyContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return DummyContext()

    def _log_performance_metrics(
        self,
        request_id: str,
        router_time: float,
        loading_time: float,
        generation_time: float,
        total_time: float
    ):
        """Log detailed performance metrics"""
        # Memory usage
        memory = psutil.virtual_memory()
        ram_used = (memory.total - memory.available) / (1024**3)

        # GPU memory if available
        gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)

        log_performance_metrics(
            self.logger,
            operation=f"request_{request_id}",
            duration=total_time,
            memory_usage=ram_used * 1024,  # Convert to MB
            gpu_memory=gpu_memory * 1024 if gpu_memory else None,  # Convert to MB
            router_time=router_time,
            loading_time=loading_time,
            generation_time=generation_time,
            request_count=self.request_count
        )

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            "initialized": self.is_initialized,
            "router_type": self.router_type,
            "total_requests": self.request_count,
            "total_inference_time": self.total_inference_time,
            "total_loading_time": self.total_loading_time,
            "average_inference_time": self.total_inference_time / max(self.request_count, 1),
            "average_loading_time": self.total_loading_time / max(self.request_count, 1),
            "current_model": self.llm_manager.current_model if self.llm_manager else None,
            "available_models": self.llm_manager.get_available_models() if self.llm_manager else [],
        }

        # Add carbon footprint if tracking enabled
        if self.carbon_tracker:
            stats["carbon_footprint"] = {
                "total_emissions_kg": self.carbon_tracker.get_total_emissions(),
                "emissions_breakdown": self.carbon_tracker.get_emissions_breakdown()
            }

        return stats

    def cleanup(self):
        """Cleanup system resources"""
        self.logger.info("Cleaning up system resources...")

        try:
            # Unload current model
            if self.llm_manager and self.llm_manager.current_model:
                self.llm_manager.unload_current_model()

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Force garbage collection
            gc.collect()

            # Log carbon footprint summary
            if self.carbon_tracker:
                self.carbon_tracker.log_summary()

            self.logger.info("System cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def switch_router(self, new_router_type: str, router_model_path: Optional[str] = None):
        """
        Switch to a different router type

        Args:
            new_router_type: New router type ("dummy", "bert", "nvidia_complexity", "graham_complexity")
            router_model_path: Path to router model (for BERT router)
        """
        self.logger.info(f"Switching router from {self.router_type} to {new_router_type}")

        self.router_type = new_router_type
        self.router_model_path = router_model_path

        # Reinitialize router
        if new_router_type == "dummy":
            self.router = DummyRouter(
                num_classes=self.config["models"]["router"]["num_classes"],
                seed=42,
                logger=self.logger
            )
        elif new_router_type == "bert":
            router_config = self.config["models"]["router"]
            self.router = BERTRouter(
                model_name=router_config["model_name"],
                num_classes=router_config["num_classes"],
                max_length=router_config["max_length"],
                logger=self.logger
            )

            if router_model_path:
                self.router.load_model(router_model_path)
        elif new_router_type == "nvidia_complexity":
            complexity_config = self.config["models"].get("complexity_routers", {}).get("nvidia", {})
            model_name = complexity_config.get("model_name", "nvidia/prompt-task-and-complexity-classifier")
            self.router = NvidiaComplexityRouter(
                model_name=model_name,
                logger=self.logger
            )
        elif new_router_type == "graham_complexity":
            complexity_config = self.config["models"].get("complexity_routers", {}).get("graham", {})
            model_name = complexity_config.get("model_name", "grahamaco/question-complexity-classifier")
            self.router = GrahamComplexityRouter(
                model_name=model_name,
                logger=self.logger
            )

        self.logger.info(f"Router switched to {new_router_type}")

    def batch_process(
        self,
        input_texts: list,
        **generation_kwargs
    ) -> list:
        """
        Process multiple requests in batch

        Args:
            input_texts: List of input texts
            **generation_kwargs: Arguments for generation

        Returns:
            List of results
        """
        self.logger.info(f"Processing batch of {len(input_texts)} requests")

        results = []
        for i, text in enumerate(input_texts):
            self.logger.info(f"Processing batch item {i+1}/{len(input_texts)}")
            result = self.predict_and_generate(text, **generation_kwargs)
            results.append(result)

        self.logger.info(f"Batch processing completed: {len(results)} results")
        return results

    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'logger'):
            self.cleanup()