"""
Direct LLM System - Single large model without routing
Dedicated system with proper carbon tracking for comparison with RouterLLM
"""

import time
import uuid
import logging
import torch
from typing import Dict, Any, Optional, Tuple
from contextlib import contextmanager

from ..models.llm_manager import LLMManager
from ..utils.carbon_tracker import CarbonTracker


class DirectLLMSystem:
    """
    Direct LLM system that uses a single large model without routing.
    Includes proper carbon tracking and performance metrics.
    """

    def __init__(
        self,
        config_path: str,
        model_name: str,
        enable_carbon_tracking: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize DirectLLMSystem

        Args:
            config_path: Path to configuration file
            model_name: Name of the LLM model to use
            enable_carbon_tracking: Whether to track carbon emissions
            logger: Logger instance
        """
        self.config_path = config_path
        self.model_name = model_name
        self.logger = logger or logging.getLogger(__name__)

        # Initialize components
        self.llm_manager = None
        self.carbon_tracker = None
        self.is_initialized = False

        # Statistics
        self.total_requests = 0
        self.total_inference_time = 0.0
        self.total_loading_time = 0.0

        # Setup carbon tracking
        if enable_carbon_tracking:
            try:
                self.carbon_tracker = CarbonTracker(
                    project_name="directllm",
                    output_dir="./logs/carbon"
                )
                self.logger.info("CarbonTracker initialized for DirectLLM")
            except Exception as e:
                self.logger.warning(f"Failed to initialize CarbonTracker: {e}")
                self.carbon_tracker = None
        else:
            self.carbon_tracker = None

        self.logger.info(f"DirectLLMSystem initialized - Model: {model_name}, Carbon tracking: {enable_carbon_tracking}")

    def process_request(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> Tuple[str, str]:
        """
        Process a request and return (response, model_used).
        This method matches the interface expected by the comparison script.
        """
        result = self.generate_response(
            input_text=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample
        )

        # Extract response text from result dict
        response = result.get('response', '') if isinstance(result, dict) else str(result)
        return response, self.model_name

    def initialize(self):
        """Initialize the system components"""
        if self.is_initialized:
            return

        self.logger.info("Initializing DirectLLM System...")

        # Initialize LLM Manager
        self.logger.info("Initializing LLM Manager...")
        self.llm_manager = LLMManager(self.config_path, logger=self.logger)

        self.is_initialized = True
        self.logger.info("DirectLLM System initialized successfully")

    @contextmanager
    def _dummy_context(self):
        """Dummy context manager when carbon tracking is disabled"""
        yield

    def generate_response(
        self,
        input_text: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> Dict[str, Any]:
        """
        Generate response using the direct LLM

        Args:
            input_text: Input text/prompt
            max_new_tokens: Maximum number of new tokens to generate
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

        self.logger.info(f"DirectLLM - Processing request {request_id}")

        try:
            # Step 1: Load Model (with carbon tracking)
            with self.carbon_tracker.track_emissions("model_loading") if self.carbon_tracker else self._dummy_context():
                loading_start = time.time()
                success = self.llm_manager.load_model(self.model_name)
                loading_time = time.time() - loading_start

                if not success:
                    raise RuntimeError(f"Failed to load model: {self.model_name}")

            self.logger.info(f"Model loaded successfully: {self.model_name}")

            # Step 2: Generate response (with carbon tracking)
            with self.carbon_tracker.track_emissions("llm_inference") if self.carbon_tracker else self._dummy_context():
                generation_start = time.time()
                response = self.llm_manager.generate_response(
                    prompt=input_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample
                )
                generation_time = time.time() - generation_start

                if response is None:
                    raise RuntimeError("Failed to generate response")

            total_time = time.time() - start_time

            # Update statistics
            self.total_requests += 1
            self.total_inference_time += generation_time
            self.total_loading_time += loading_time

            # Log performance metrics
            memory_usage = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            gpu_memory = torch.cuda.memory_reserved() / (1024**2) if torch.cuda.is_available() else 0

            self.logger.info(f"PERFORMANCE_METRICS | operation: request_{request_id} | "
                           f"duration_seconds: {total_time:.4f} | memory_usage_mb: {memory_usage:.2f} | "
                           f"gpu_memory_mb: {gpu_memory:.2f} | loading_time: {loading_time} | "
                           f"generation_time: {generation_time} | request_count: {self.total_requests}")

            self.logger.info(f"REQUEST_RESPONSE | request_id: {request_id} | "
                           f"input: {input_text[:100]}... | model_used: {self.model_name} | "
                           f"response_length: {len(response)} | total_time: {total_time:.4f}")

            self.logger.info(f"Request {request_id} completed successfully in {total_time:.4f}s")

            return {
                "response": response,
                "model_used": self.model_name,
                "request_id": request_id,
                "total_time": total_time,
                "loading_time": loading_time,
                "generation_time": generation_time,
                "input_length": len(input_text),
                "response_length": len(response),
                "status": "success"
            }

        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"DirectLLM request failed: {e}")

            return {
                "response": None,
                "model_used": self.model_name,
                "request_id": request_id,
                "total_time": total_time,
                "loading_time": 0.0,
                "generation_time": 0.0,
                "input_length": len(input_text),
                "response_length": 0,
                "status": "error",
                "error": str(e)
            }

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics and carbon footprint"""
        stats = {
            "initialized": self.is_initialized,
            "model_name": self.model_name,
            "total_requests": self.total_requests,
            "total_inference_time": self.total_inference_time,
            "total_loading_time": self.total_loading_time,
            "average_inference_time": (
                self.total_inference_time / self.total_requests if self.total_requests > 0 else 0.0
            ),
            "average_loading_time": (
                self.total_loading_time / self.total_requests if self.total_requests > 0 else 0.0
            ),
            "current_model": self.llm_manager.current_model if self.llm_manager else None,
        }

        # Add carbon footprint if tracking is enabled
        if self.carbon_tracker:
            try:
                total_emissions = self.carbon_tracker.get_total_emissions()
                emissions_breakdown = self.carbon_tracker.get_emissions_breakdown()

                carbon_stats = {
                    "total_emissions_kg": total_emissions,
                    "emissions_breakdown": emissions_breakdown
                }
                stats["carbon_footprint"] = carbon_stats

                # Log carbon summary
                self.logger.info(f"Total emissions across all components: {total_emissions:.6f} kg CO2")

            except Exception as e:
                self.logger.warning(f"Failed to get carbon stats: {e}")
                stats["carbon_footprint"] = {"total_emissions_kg": 0, "emissions_breakdown": {}}
        else:
            stats["carbon_footprint"] = {"total_emissions_kg": 0, "emissions_breakdown": {}}

        return stats

    def cleanup(self):
        """Clean up system resources"""
        self.logger.info("Cleaning up DirectLLM system resources...")

        if self.llm_manager:
            self.llm_manager.unload_current_model()

        # Final carbon footprint summary
        if self.carbon_tracker:
            try:
                carbon_stats = self.carbon_tracker.get_system_stats()
                total_emissions = carbon_stats.get("carbon_footprint", {}).get("total_emissions_kg", 0)

                self.logger.info("============================================================")
                self.logger.info("CARBON FOOTPRINT SUMMARY - DirectLLM")
                self.logger.info("============================================================")

                emissions_breakdown = carbon_stats.get("carbon_footprint", {}).get("emissions_breakdown", {})
                for component, emissions in emissions_breakdown.items():
                    percentage = (emissions / total_emissions * 100) if total_emissions > 0 else 0
                    self.logger.info(f"{component}: {emissions:.6f} kg CO2 ({percentage:.1f}%)")

                self.logger.info("------------------------------------------------------------")
                self.logger.info(f"TOTAL: {total_emissions:.6f} kg CO2")
                self.logger.info(f"Equivalent to driving ~{total_emissions * 4000:.2f} km in a car")
                self.logger.info("============================================================")

            except Exception as e:
                self.logger.warning(f"Failed to get final carbon stats: {e}")

        self.logger.info("DirectLLM system cleanup completed")