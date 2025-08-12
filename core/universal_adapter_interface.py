"""
Universal Adapter Interface

This module defines the interface that user adapters must implement to work
with the engine's universal benchmark system. Users create adapters that
map their models to the engine's standardized format.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np


class UniversalAdapterInterface(ABC):
    """
    Universal adapter interface that users must implement.
    
    This interface ensures that user adapters can work with the engine's
    standardized format regardless of their underlying model architecture.
    """
    
    @abstractmethod
    def get_standardized_input_shape(self) -> tuple:
        """
        Return the input shape that matches the engine's standard.
        
        Returns:
            Tuple representing the input shape (e.g., (1, 512))
        """
        pass
    
    @abstractmethod
    def get_standardized_output_shape(self) -> tuple:
        """
        Return the output shape that matches the engine's standard.
        
        Returns:
            Tuple representing the output shape (e.g., (1, 24))
        """
        pass
    
    @abstractmethod
    def preprocess_for_standard(self, raw_input: Union[str, Dict[str, Any]]) -> np.ndarray:
        """
        Preprocess input to match the engine's standardized format.
        
        Args:
            raw_input: Raw input (text or dict containing text)
            
        Returns:
            Preprocessed input as numpy array matching standardized shape
        """
        pass
    
    @abstractmethod
    def postprocess_from_standard(self, model_output: np.ndarray) -> np.ndarray:
        """
        Postprocess model output to match the engine's standardized format.
        
        Args:
            model_output: Raw model output
            
        Returns:
            Postprocessed output as numpy array matching standardized shape
        """
        pass
    
    @abstractmethod
    def validate_standard_compatibility(self, standard_config: Dict[str, Any]) -> bool:
        """
        Validate that this adapter is compatible with the engine's standard.
        
        Args:
            standard_config: Engine's standardized configuration
            
        Returns:
            True if compatible, False otherwise
        """
        pass
    
    @abstractmethod
    def get_adapter_info(self) -> Dict[str, Any]:
        """
        Get information about this adapter.
        
        Returns:
            Dictionary containing adapter metadata
        """
        pass


class StandardizedEmotionAdapter(UniversalAdapterInterface):
    """
    Example implementation of a standardized emotion adapter.
    
    Users should inherit from this class and implement the abstract methods
    to make their models work with the universal benchmark system.
    """
    
    def __init__(self, model_path: str, standard_config: Dict[str, Any]):
        """
        Initialize the standardized adapter.
        
        Args:
            model_path: Path to the user's model
            standard_config: Engine's standardized configuration
        """
        self.model_path = model_path
        self.standard_config = standard_config
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the user's model. Override this method."""
        raise NotImplementedError("Users must implement model loading")
    
    def get_standardized_input_shape(self) -> tuple:
        """Return standardized input shape."""
        return self.standard_config["input_shape"]
    
    def get_standardized_output_shape(self) -> tuple:
        """Return standardized output shape."""
        return (1, self.standard_config["num_classes"])
    
    def preprocess_for_standard(self, raw_input: Union[str, Dict[str, Any]]) -> np.ndarray:
        """
        Preprocess input to match engine standard.
        
        Users must implement this to convert their model's input format
        to the engine's standardized format.
        """
        raise NotImplementedError("Users must implement preprocessing")
    
    def postprocess_from_standard(self, model_output: np.ndarray) -> np.ndarray:
        """
        Postprocess output to match engine standard.
        
        Users must implement this to convert their model's output format
        to the engine's standardized format.
        """
        raise NotImplementedError("Users must implement postprocessing")
    
    def validate_standard_compatibility(self, standard_config: Dict[str, Any]) -> bool:
        """Validate compatibility with engine standard."""
        try:
            # Check if input/output shapes are compatible
            input_shape = self.get_standardized_input_shape()
            output_shape = self.get_standardized_output_shape()
            
            # Validate input shape
            if input_shape != standard_config["input_shape"]:
                return False
            
            # Validate output shape
            if output_shape[1] != standard_config["num_classes"]:
                return False
            
            return True
        except Exception:
            return False
    
    def get_adapter_info(self) -> Dict[str, Any]:
        """Get adapter information."""
        return {
            "type": "StandardizedEmotionAdapter",
            "model_path": self.model_path,
            "input_shape": self.get_standardized_input_shape(),
            "output_shape": self.get_standardized_output_shape(),
            "standardized": True
        }
    
    def run_inference(self, preprocessed_input: np.ndarray) -> np.ndarray:
        """
        Run inference on preprocessed input.
        
        Users must implement this to run their model.
        """
        raise NotImplementedError("Users must implement inference")


# Example usage for users:
"""
To use the universal benchmark system, users must:

1. Create an adapter that inherits from StandardizedEmotionAdapter
2. Implement the abstract methods:
   - load_model(): Load their specific model
   - preprocess_for_standard(): Convert input to engine's format
   - postprocess_from_standard(): Convert output to engine's format
   - run_inference(): Run their model

Example:

class MyTFLiteAdapter(StandardizedEmotionAdapter):
    def load_model(self):
        # Load your TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        
    def preprocess_for_standard(self, raw_input):
        # Convert text to your model's input format
        # Then reshape to match engine's standard
        return standardized_input
        
    def postprocess_from_standard(self, model_output):
        # Convert your model's output to engine's standard
        return standardized_output
        
    def run_inference(self, preprocessed_input):
        # Run your TFLite model
        return model_output

# Then use it:
adapter = MyTFLiteAdapter("my_model.tflite", standard_config)
if adapter.validate_standard_compatibility(standard_config):
    # Use with universal benchmark
    results = engine.run_universal_benchmark(adapter)
"""
