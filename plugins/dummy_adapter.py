"""
Dummy model adapter for testing the BenchmarkEngine framework.

This adapter simulates model loading and inference without requiring
actual model files or ML frameworks.
"""

import time
import random
from typing import Dict, Any
from core import BaseModelAdapter, ModelType


class DummyModelAdapter(BaseModelAdapter):
    """
    A dummy model adapter that simulates model behavior.

    This is useful for testing the framework without requiring
    actual model files or ML frameworks.
    """

    def __init__(self):
        self.model_loaded = False
        self.model_name = "DummyModel"
        self.model_size = 0
        self.inference_delay = 0.01  # Simulate 10ms inference time
        self.config = {}

    def load(self, model_path: str) -> bool:
        """
        Simulate loading a model.

        Args:
            model_path: Path to the model file (ignored in dummy implementation)

        Returns:
            True if "loaded" successfully
        """
        print(f"Loading dummy model from {model_path}...")
        time.sleep(0.1)  # Simulate loading time

        # Simulate model metadata
        self.model_name = f"DummyModel-{random.randint(1000, 9999)}"
        self.model_size = random.randint(10, 100)  # MB
        self.model_loaded = True

        print(f"Dummy model loaded: {self.model_name} ({self.model_size}MB)")
        return True

    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Configure model parameters.

        Args:
            config: Configuration dictionary with parameters like batch_size, precision, device

        Returns:
            True if configured successfully
        """
        print(f"Configuring dummy model with: {config}")

        # Update configuration
        self.config.update(config)

        # Simulate configuration effects
        if "precision" in config:
            if config["precision"] == "fp16":
                self.inference_delay = 0.008  # Faster with fp16
            elif config["precision"] == "int8":
                self.inference_delay = 0.006  # Even faster with int8

        if "device" in config and config["device"] == "gpu":
            self.inference_delay = 0.005  # Faster on GPU

        print(f" Dummy model configured")
        return True

    def preprocess_input(self, sample: Any) -> Any:
        """
        Convert dataset sample to model input format.

        Args:
            sample: Raw sample from dataset

        Returns:
            Preprocessed input for model
        """
        # In a real implementation, this would:
        # - Resize images
        # - Normalize pixel values
        # - Tokenize text
        # - Convert to tensors

        if isinstance(sample, dict):
            # Handle structured samples
            if "features" in sample:
                # Simulate feature preprocessing
                return {"input": sample["features"]}
            elif "image" in sample:
                # Simulate image preprocessing
                return {"input": sample["image"]}
            elif "text" in sample:
                # Simulate text preprocessing
                return {"input": sample["text"]}

        # Default: return as-is
        return sample

    def run(self, inputs: Any) -> Any:
        """
        Simulate model inference.

        Args:
            inputs: Preprocessed input data

        Returns:
            Raw model output
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Simulate inference time
        time.sleep(self.inference_delay)

        # Generate dummy prediction
        if isinstance(inputs, (list, tuple)) and len(inputs) > 0:
            # If input is a sequence, return a sequence of predictions
            return [
                f"raw_prediction_{i}_{random.randint(1, 100)}"
                for i in range(len(inputs))
            ]
        else:
            # Single prediction
            return f"raw_prediction_{random.randint(1, 100)}"

    def postprocess_output(self, model_output: Any) -> Any:
        """
        Convert model output to standardized format.

        Args:
            model_output: Raw model output

        Returns:
            Standardized prediction
        """
        # In a real implementation, this would:
        # - Apply softmax for classification
        # - Decode tokens for text generation
        # - Convert to human-readable format

        if isinstance(model_output, (list, tuple)):
            # Convert raw predictions to standardized format
            return [
                f"prediction_{i}_{random.randint(1, 100)}"
                for i in range(len(model_output))
            ]
        else:
            # Single prediction
            return f"prediction_{random.randint(1, 100)}"

    def get_model_info(self) -> Dict[str, Any]:
        """
        Return metadata about the loaded model.

        Returns:
            Dictionary containing model information
        """
        return {
            "name": self.model_name,
            "type": "dummy",
            "size_mb": self.model_size,
            "loaded": self.model_loaded,
            "inference_delay_ms": self.inference_delay * 1000,
            "config": self.config,
        }

    def get_model_type(self) -> ModelType:
        """
        Return the type of model for validation.

        Returns:
            ModelType enum value
        """
        return ModelType.CUSTOM
