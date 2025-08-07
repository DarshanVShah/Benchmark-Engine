"""
Abstract interfaces for the BenchmarkEngine framework.

This module defines the base classes that all adapters, metrics, and datasets must implement.
Uses Template Method pattern with explicit data contracts for type safety.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Union
from enum import Enum
from .types import ModelType


class DataType(Enum):
    """Standardized data types for input/output contracts."""

    TEXT = "text"
    IMAGE = "image"
    TENSOR = "tensor"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


class OutputType(Enum):
    """Standardized output types for model predictions."""

    CLASS_ID = "class_id"
    PROBABILITIES = "probabilities"
    LOGITS = "logits"
    TOKENS = "tokens"
    EMBEDDINGS = "embeddings"
    REGRESSION = "regression"


class BaseModelAdapter(ABC):
    """
    Abstract interface that all model adapters must implement.

    Uses Template Method pattern to define the complete inference pipeline:
    raw_input → preprocess → run → postprocess → standardized_output
    """

    @property
    @abstractmethod
    def input_type(self) -> DataType:
        """Return the expected input type for this model."""
        pass

    @property
    @abstractmethod
    def output_type(self) -> OutputType:
        """Return the output type this model produces."""
        pass

    @abstractmethod
    def load(self, model_path: str) -> bool:
        """Load the model from the given path."""
        pass

    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure model parameters (batch_size, precision, device, etc.)."""
        pass

    @abstractmethod
    def preprocess_input(self, raw_input: Any) -> Any:
        """
        Convert raw input to model-specific format.

        Args:
            raw_input: Raw input from dataset (e.g., text, image, tensor)

        Returns:
            Model-specific input format
        """
        pass

    @abstractmethod
    def run(self, preprocessed_input: Any) -> Any:
        """
        Run inference on preprocessed input.

        Args:
            preprocessed_input: Input in model-specific format

        Returns:
            Raw model output
        """
        pass

    @abstractmethod
    def postprocess_output(self, model_output: Any) -> Any:
        """
        Convert raw model output to standardized format.

        Args:
            model_output: Raw output from model

        Returns:
            Standardized output matching output_type
        """
        pass

    def predict(self, raw_input: Any) -> Any:
        """
        Complete inference pipeline using Template Method pattern.

        Args:
            raw_input: Raw input from dataset

        Returns:
            Standardized prediction
        """
        preprocessed = self.preprocess_input(raw_input)
        model_output = self.run(preprocessed)
        return self.postprocess_output(model_output)

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded model."""
        pass

    @abstractmethod
    def get_model_type(self) -> ModelType:
        """Return the type of model (for validation)."""
        pass

    def validate_compatibility(self, dataset: "BaseDataset") -> bool:
        """
        Validate that this model is compatible with the given dataset.

        Args:
            dataset: Dataset to validate against

        Returns:
            True if compatible, False otherwise
        """
        return dataset.output_type == self.input_type


class BaseMetric(ABC):
    """Abstract interface that all metrics must implement."""

    @property
    @abstractmethod
    def expected_input_type(self) -> OutputType:
        """Return the expected model output type for this metric."""
        pass

    @abstractmethod
    def calculate(
        self, predictions: List[Any], targets: List[Any], **kwargs
    ) -> Dict[str, float]:
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

    @property
    @abstractmethod
    def output_type(self) -> DataType:
        """Return the type of data this dataset produces."""
        pass

    @abstractmethod
    def load(self, dataset_path: str) -> bool:
        """Load the dataset from the given path."""
        pass

    @abstractmethod
    def get_samples(self, num_samples: Optional[int] = None) -> List[Any]:
        """Get samples from the dataset (for compatibility)."""
        pass

    @abstractmethod
    def get_samples_with_targets(
        self, num_samples: Optional[int] = None
    ) -> List[Tuple[Any, Any]]:
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
