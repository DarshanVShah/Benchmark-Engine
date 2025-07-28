"""
Dummy model adapter for testing the BenchmarkEngine framework.

This adapter simulates model loading and inference without requiring
actual model files or ML frameworks.
"""

import time
import random
from typing import Dict, Any
from core import BaseModelAdapter


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
        
        print(f"✓ Dummy model loaded: {self.model_name} ({self.model_size}MB)")
        return True
    
    def run(self, inputs: Any) -> Any:
        """
        Simulate model inference.
        
        Args:
            inputs: Input data (can be any type for dummy implementation)
            
        Returns:
            Simulated prediction output
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Simulate inference time
        time.sleep(self.inference_delay)
        
        # Generate dummy prediction
        if isinstance(inputs, (list, tuple)) and len(inputs) > 0:
            # If input is a sequence, return a sequence of predictions
            return [f"prediction_{i}_{random.randint(1, 100)}" for i in range(len(inputs))]
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
            "inference_delay_ms": self.inference_delay * 1000
        } 