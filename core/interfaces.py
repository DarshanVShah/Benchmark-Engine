"""
Abstract interfaces for the BenchmarkEngine framework.

This module defines the base classes that all adapters, metrics, and datasets must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
from .types import ModelType


class BaseModelAdapter(ABC):
    """Abstract interface that all model adapters must implement."""
    
    @abstractmethod
    def load(self, model_path: str) -> bool:
        """Load the model from the given path."""
        pass
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure model parameters (batch_size, precision, device, etc.)."""
        pass
    
    @abstractmethod
    def preprocess_input(self, sample: Any) -> Any:
        """Convert dataset sample to model input format."""
        pass
    
    @abstractmethod
    def run(self, inputs: Any) -> Any:
        """Run inference on the given inputs."""
        pass
    
    @abstractmethod
    def postprocess_output(self, model_output: Any) -> Any:
        """Convert model output to standardized format."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded model."""
        pass
    
    @abstractmethod
    def get_model_type(self) -> ModelType:
        """Return the type of model (for validation)."""
        pass


class BaseMetric(ABC):
    """Abstract interface that all metrics must implement."""
    
    @abstractmethod
    def calculate(self, predictions: List[Any], targets: List[Any], **kwargs) -> Dict[str, float]:
        """Calculate metric values given predictions and targets."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this metric."""
        pass
    
    @abstractmethod
    def validate_inputs(self, predictions: List[Any], targets: List[Any]) -> bool:
        """Validate that inputs are compatible with this metric."""
        pass


class BaseDataset(ABC):
    """Abstract interface that all datasets must implement."""
    
    @abstractmethod
    def load(self, dataset_path: str) -> bool:
        """Load the dataset from the given path."""
        pass
    
    @abstractmethod
    def get_samples(self, num_samples: Optional[int] = None) -> List[Any]:
        """Get samples from the dataset (for compatibility)."""
        pass
    
    @abstractmethod
    def get_samples_with_targets(self, num_samples: Optional[int] = None) -> List[Tuple[Any, Any]]:
        """Get (input, target) pairs for evaluation."""
        pass
    
    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded dataset."""
        pass
    
    @abstractmethod
    def get_input_shape(self) -> Tuple[int, ...]:
        """Return the expected input shape for models."""
        pass 