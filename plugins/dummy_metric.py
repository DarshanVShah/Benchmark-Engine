"""
Dummy accuracy metric for testing the BenchmarkEngine framework.

This metric simulates accuracy calculation without requiring
actual ground truth data.
"""

import random
from typing import Dict, Any
from core import BaseMetric


class DummyAccuracyMetric(BaseMetric):
    """
    A dummy accuracy metric that simulates accuracy calculation.
    
    This is useful for testing the framework without requiring
    actual ground truth data.
    """
    
    def __init__(self):
        self.metric_name = "DummyAccuracy"
        
    def calculate(self, predictions: Any, targets: Any, **kwargs) -> Dict[str, float]:
        """
        Calculate dummy accuracy metrics.
        
        Args:
            predictions: Model predictions (ignored in dummy implementation)
            targets: Ground truth targets (ignored in dummy implementation)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing accuracy metrics
        """
        # Simulate accuracy calculation
        num_predictions = len(predictions) if hasattr(predictions, '__len__') else 1
        
        # Generate realistic-looking accuracy metrics
        base_accuracy = random.uniform(0.7, 0.95)  # 70-95% accuracy
        precision = base_accuracy + random.uniform(-0.1, 0.1)
        recall = base_accuracy + random.uniform(-0.1, 0.1)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "accuracy": round(base_accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "num_samples": num_predictions
        }
    
    def get_name(self) -> str:
        """
        Return the name of this metric.
        
        Returns:
            Metric name
        """
        return self.metric_name 