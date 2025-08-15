"""
Dummy model adapter for testing the BenchmarkEngine framework.

This adapter simulates model loading and inference without requiring
actual model files or ML frameworks.
"""

import random
import time
from typing import Any, Dict

from core.interfaces import BaseModelAdapter, DataType, ModelType, OutputType


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

    @property
    def input_type(self) -> DataType:
        """Dummy model accepts text input."""
        return DataType.TEXT

    @property
    def output_type(self) -> OutputType:
        """Dummy model outputs class IDs."""
        return OutputType.CLASS_ID

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

        print("Dummy model configured")
        return True

    def preprocess(self, raw_input: Any) -> Any:
        """
        Convert raw input to model-specific format.

        Args:
            raw_input: Raw input from dataset

        Returns:
            Preprocessed input for model
        """
        # In a real implementation, this would:
        # - Resize images
        # - Normalize pixel values
        # - Tokenize text
        # - Convert to tensors

        if isinstance(raw_input, dict):
            # Handle structured samples
            if "features" in raw_input:
                # Simulate feature preprocessing
                return {"input": raw_input["features"]}
            elif "image" in raw_input:
                # Simulate image preprocessing
                return {"input": raw_input["image"]}
            else:
                # Generic preprocessing
                return {"input": raw_input}
        elif isinstance(raw_input, str):
            # Simulate text preprocessing
            return {"input": raw_input.lower().strip()}
        else:
            # Generic preprocessing
            return {"input": raw_input}

    def run(self, preprocessed_input: Any) -> Any:
        """
        Run inference on preprocessed input.

        Args:
            preprocessed_input: Preprocessed input from preprocess()

        Returns:
            Raw model output
        """
        if not self.model_loaded:
            print("Error: Model not loaded")
            return None

        # Simulate inference time
        time.sleep(self.inference_delay)

        # Simulate model output
        # In a real implementation, this would be actual model inference
        if isinstance(preprocessed_input, dict) and "input" in preprocessed_input:
            input_data = preprocessed_input["input"]

            # Simple rule-based classification for demonstration
            if isinstance(input_data, str):
                # Text classification simulation
                positive_words = ["good", "great", "excellent", "amazing", "wonderful"]
                negative_words = ["bad", "terrible", "awful", "horrible", "poor"]

                text = input_data.lower()
                positive_score = sum(1 for word in positive_words if word in text)
                negative_score = sum(1 for word in negative_words if word in text)

                if positive_score > negative_score:
                    return [1]  # Positive class
                elif negative_score > positive_score:
                    return [0]  # Negative class
                else:
                    return [random.randint(0, 1)]  # Random for neutral
            else:
                # Generic classification
                return [random.randint(0, 2)]  # Random class
        else:
            # Fallback
            return [random.randint(0, 1)]

    def postprocess(self, model_output: Any) -> Any:
        """
        Convert raw model output to standardized format.

        Args:
            model_output: Raw output from model

        Returns:
            Standardized prediction
        """
        if model_output is None:
            return None

        # Ensure output is in the expected format (list of class IDs)
        if isinstance(model_output, list):
            return model_output
        elif isinstance(model_output, (int, float)):
            return [int(model_output)]
        else:
            return [0]  # Default fallback

    def get_model_info(self) -> Dict[str, Any]:
        """
        Return metadata about the loaded model.

        Returns:
            Dictionary containing model information
        """
        info = {
            "name": self.model_name,
            "type": "dummy",
            "model_size_mb": self.model_size,
            "loaded": self.model_loaded,
            "inference_delay": self.inference_delay,
            "config": self.config,
        }

        return info

    def get_model_type(self) -> ModelType:
        """
        Return the type of model for validation.

        Returns:
            ModelType enum value
        """
        return ModelType.CUSTOM
