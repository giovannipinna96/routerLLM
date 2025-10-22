"""
Logging utilities for RouterLLM
"""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "routerllm",
    log_dir: str = "./logs",
    log_file: str = "routerllm.log",
    level: str = "INFO"
) -> logging.Logger:
    """
    Set up a logger with file and console handlers

    Args:
        name: Logger name
        log_dir: Directory to save log files
        log_file: Log file name
        level: Logging level

    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    # File handler
    log_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(getattr(logging, level.upper()))
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logger initialized. Log file: {log_path}")
    return logger


def log_performance_metrics(
    logger: logging.Logger,
    operation: str,
    duration: float,
    memory_usage: Optional[float] = None,
    gpu_memory: Optional[float] = None,
    **kwargs
):
    """
    Log performance metrics in a structured format

    Args:
        logger: Logger instance
        operation: Name of the operation
        duration: Duration in seconds
        memory_usage: Memory usage in MB
        gpu_memory: GPU memory usage in MB
        **kwargs: Additional metrics
    """
    metrics = {
        "operation": operation,
        "duration_seconds": round(duration, 4),
    }

    if memory_usage is not None:
        metrics["memory_usage_mb"] = round(memory_usage, 2)

    if gpu_memory is not None:
        metrics["gpu_memory_mb"] = round(gpu_memory, 2)

    # Add any additional metrics
    metrics.update(kwargs)

    # Create a formatted string
    metric_str = " | ".join([f"{k}: {v}" for k, v in metrics.items()])
    logger.info(f"PERFORMANCE_METRICS | {metric_str}")


def log_model_info(logger: logging.Logger, model_name: str, model_size: str, parameters: int):
    """
    Log model information

    Args:
        logger: Logger instance
        model_name: Name of the model
        model_size: Size description (e.g., "7B")
        parameters: Number of parameters
    """
    logger.info(f"MODEL_INFO | name: {model_name} | size: {model_size} | parameters: {parameters:,}")


def log_request_response(
    logger: logging.Logger,
    request_id: str,
    input_text: str,
    predicted_model: str,
    response: str,
    confidence: Optional[float] = None
):
    """
    Log request and response information

    Args:
        logger: Logger instance
        request_id: Unique request identifier
        input_text: Input text (truncated for logging)
        predicted_model: Model selected by router
        response: Generated response (truncated for logging)
        confidence: Router confidence score
    """
    # Truncate long texts for logging
    input_truncated = input_text[:100] + "..." if len(input_text) > 100 else input_text
    response_truncated = response[:200] + "..." if len(response) > 200 else response

    log_data = {
        "request_id": request_id,
        "input": input_truncated,
        "predicted_model": predicted_model,
        "response_length": len(response),
    }

    if confidence is not None:
        log_data["confidence"] = round(confidence, 4)

    log_str = " | ".join([f"{k}: {v}" for k, v in log_data.items()])
    logger.info(f"REQUEST_RESPONSE | {log_str}")