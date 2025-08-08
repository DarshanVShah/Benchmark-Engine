"""
Simple Accuracy Metric for the refactored framework.

This metric demonstrates the new architecture where:
- Metric declares its expected_input_type (CLASS_ID)
- Model adapters must declare their output_type (CLASS_ID)
- Framework validates compatibility before running
"""

from typing import Any, Dict, List
from core import BaseMetric, OutputType


class SimpleAccuracyMetric(BaseMetric):
    """
    Simple accuracy metric for classification tasks.
    
    This metric expects CLASS_ID outputs from models and calculates accuracy
    by comparing predicted class IDs with target class IDs.
    """

    def __init__(self):
        self.name = "SimpleAccuracy"

    @property
    def expected_input_type(self) -> OutputType:
        """This metric expects class ID predictions."""
        return OutputType.CLASS_ID

    def calculate(self, predictions: List[Any], targets: List[Any], **kwargs) -> Dict[str, float]:
        """
        Calculate accuracy given predictions and targets.
        
        Args:
            predictions: List of predicted class IDs
            targets: List of target class IDs
            **kwargs: Additional arguments (not used)
            
        Returns:
            Dictionary with accuracy score
        """
        if not predictions or not targets:
            return {"accuracy": 0.0}
        
        if len(predictions) != len(targets):
            print(f"Warning: Mismatch between predictions ({len(predictions)}) and targets ({len(targets)})")
            return {"accuracy": 0.0}
        
        # Count correct predictions
        correct = 0
        total = 0
        
        for pred, target in zip(predictions, targets):
            # Handle different prediction formats
            if isinstance(pred, list):
                # If prediction is a list, take the first element
                pred_class = pred[0] if pred else None
            elif isinstance(pred, (int, float)):
                # If prediction is a number, use it directly
                pred_class = int(pred)
            else:
                # Skip invalid predictions
                continue
            
            if pred_class is not None and pred_class == target:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {"accuracy": accuracy}

    def get_name(self) -> str:
        """Return the name of this metric."""
        return self.name

    def validate_inputs(self, predictions: List[Any], targets: List[Any]) -> bool:
        """
        Validate that inputs are compatible with this metric.
        
        Args:
            predictions: List of model predictions
            targets: List of target values
            
        Returns:
            True if inputs are valid, False otherwise
        """
        if not predictions or not targets:
            return False
        
        if len(predictions) != len(targets):
            return False
        
        # Check that predictions are valid (not None, not empty lists)
        for pred in predictions:
            if pred is None:
                return False
            if isinstance(pred, list) and len(pred) == 0:
                return False
        
        return True
