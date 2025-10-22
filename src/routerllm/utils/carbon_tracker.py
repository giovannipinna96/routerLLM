"""
Carbon footprint tracking utilities using CodeCarbon
"""

import os
import logging
import warnings
from typing import Optional, Dict, Any
from contextlib import contextmanager

# Suppress CodeCarbon warnings about multiple instances - this is intentional for our architecture
warnings.filterwarnings("ignore", message="Multiple instances of codecarbon are allowed to run at the same time")
warnings.filterwarnings("ignore", message="No CPU tracking mode found")
warnings.filterwarnings("ignore", message="Please ensure RAPL files exist")

from codecarbon import EmissionsTracker, OfflineEmissionsTracker


class CarbonTracker:
    """
    Carbon footprint tracker for RouterLLM operations
    """

    def __init__(
        self,
        project_name: str = "routerllm",
        output_dir: str = "./logs/carbon",
        country_iso_code: str = "USA",
        offline_mode: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize carbon tracker

        Args:
            project_name: Name of the project for tracking
            output_dir: Directory to save emission logs
            country_iso_code: ISO code for carbon intensity calculation
            offline_mode: Use offline mode if no internet connection
            logger: Logger instance
        """
        self.project_name = project_name
        self.output_dir = output_dir
        self.country_iso_code = country_iso_code
        self.offline_mode = offline_mode
        self.logger = logger or logging.getLogger(__name__)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Track emissions by component
        self.component_emissions: Dict[str, float] = {}
        self.active_trackers: Dict[str, Any] = {}

        self.logger.info(f"CarbonTracker initialized - Project: {project_name}, Output: {output_dir}")

    def create_tracker(self, task_name: str) -> Any:
        """
        Create a new emissions tracker for a specific task

        Args:
            task_name: Name of the task being tracked

        Returns:
            EmissionsTracker instance
        """
        if self.offline_mode:
            tracker = OfflineEmissionsTracker(
                project_name=f"{self.project_name}_{task_name}",
                output_dir=self.output_dir,
                country_iso_code=self.country_iso_code
            )
        else:
            tracker = EmissionsTracker(
                project_name=f"{self.project_name}_{task_name}",
                output_dir=self.output_dir
            )

        self.logger.info(f"Created carbon tracker for task: {task_name}")
        return tracker

    @contextmanager
    def track_emissions(self, component_name: str):
        """
        Context manager for tracking emissions of a specific component

        Args:
            component_name: Name of the component being tracked

        Usage:
            with carbon_tracker.track_emissions("router_inference"):
                # Your code here
                result = router.predict(input_text)
        """
        tracker = self.create_tracker(component_name)

        try:
            self.logger.info(f"Starting carbon tracking for: {component_name}")
            tracker.start()
            self.active_trackers[component_name] = tracker
            yield tracker

        finally:
            if component_name in self.active_trackers:
                emissions = tracker.stop()
                del self.active_trackers[component_name]

                if emissions is not None:
                    self.component_emissions[component_name] = emissions
                    self.logger.info(
                        f"Carbon tracking completed for {component_name}: "
                        f"{emissions:.6f} kg CO2"
                    )
                    self._log_carbon_metrics(component_name, emissions)
                else:
                    self.logger.warning(f"No emissions data recorded for {component_name}")

    def get_total_emissions(self) -> float:
        """
        Get total emissions across all tracked components

        Returns:
            Total emissions in kg CO2
        """
        total = sum(self.component_emissions.values())
        self.logger.info(f"Total emissions across all components: {total:.6f} kg CO2")
        return total

    def get_component_emissions(self, component_name: str) -> float:
        """
        Get emissions for a specific component

        Args:
            component_name: Name of the component

        Returns:
            Emissions in kg CO2
        """
        return self.component_emissions.get(component_name, 0.0)

    def get_emissions_breakdown(self) -> Dict[str, float]:
        """
        Get breakdown of emissions by component

        Returns:
            Dictionary mapping component names to emissions
        """
        return self.component_emissions.copy()

    def reset_emissions(self):
        """Reset all tracked emissions"""
        self.component_emissions.clear()
        self.logger.info("Reset all tracked emissions")

    def _log_carbon_metrics(self, component: str, emissions: float):
        """
        Log carbon metrics in a structured format

        Args:
            component: Component name
            emissions: Emissions in kg CO2
        """
        # Convert to different units for better understanding
        emissions_g = emissions * 1000  # grams

        # Rough equivalents for context
        km_car = emissions * 4.6  # Approximate km in a average car

        self.logger.info(
            f"CARBON_METRICS | component: {component} | "
            f"emissions_kg_co2: {emissions:.6f} | "
            f"emissions_g_co2: {emissions_g:.3f} | "
            f"equivalent_car_km: {km_car:.3f}"
        )

    def log_summary(self):
        """Log a summary of all tracked emissions"""
        total_emissions = self.get_total_emissions()

        self.logger.info("=" * 60)
        self.logger.info("CARBON FOOTPRINT SUMMARY")
        self.logger.info("=" * 60)

        for component, emissions in self.component_emissions.items():
            percentage = (emissions / total_emissions * 100) if total_emissions > 0 else 0
            self.logger.info(f"{component}: {emissions:.6f} kg CO2 ({percentage:.1f}%)")

        self.logger.info("-" * 60)
        self.logger.info(f"TOTAL: {total_emissions:.6f} kg CO2")
        self.logger.info(f"Equivalent to driving ~{total_emissions * 4.6:.2f} km in a car")
        self.logger.info("=" * 60)