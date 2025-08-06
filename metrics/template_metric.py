"""
Template Metric Pattern

This module provides a clean template pattern for metrics.
Users explicitly select the metric type - no auto-detection.
"""

import numpy as np
from typing import Dict, Any, List
from core import BaseMetric, OutputType


class TemplateAccuracyMetric(BaseMetric):
    """
    Template accuracy metric for classification tasks.
    
    Users specify:
    - Input type (class_id, probabilities)
    - Threshold (for probability inputs)
    - Task type (classification, multi-label)
    """
    
    def __init__(self, input_type: str = "class_id", threshold: float = 0.5):
        self.input_type = input_type
        self.threshold = threshold
        self.name = "TemplateAccuracy"
        
    @property
    def expected_input_type(self) -> OutputType:
        """Expect the configured input type."""
        if self.input_type == "class_id":
            return OutputType.CLASS_ID
        elif self.input_type == "probabilities":
            return OutputType.PROBABILITIES
        else:
            return OutputType.CLASS_ID
    
    def get_name(self) -> str:
        """Return metric name."""
        return self.name
    
    def validate_inputs(self, predictions: List[Any], targets: List[Any]) -> bool:
        """Validate that inputs are compatible."""
        if not predictions or not targets:
            return False
        
        # Check if predictions match expected type
        if self.input_type == "probabilities":
            if not isinstance(predictions[0], (list, np.ndarray)):
                return False
        
        return True
    
    def calculate(self, predictions: List[Any], targets: List[Any]) -> Dict[str, Any]:
        """
        Calculate accuracy based on configured input type.
        """
        if not self.validate_inputs(predictions, targets):
            return {"error": "Invalid inputs for template accuracy"}
        
        # Convert predictions to class IDs if needed
        if self.input_type == "probabilities":
            class_predictions = []
            for pred in predictions:
                if isinstance(pred, list):
                    pred_array = np.array(pred)
                else:
                    pred_array = pred
                
                # Convert to class ID using threshold
                class_pred = int(np.argmax(pred_array))
                class_predictions.append(class_pred)
        else:
            class_predictions = predictions
        
        # Calculate accuracy
        correct = 0
        total = len(class_predictions)
        
        for pred, target in zip(class_predictions, targets):
            if pred == target:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "accuracy": float(accuracy),
            "correct": correct,
            "total": total
        }


class TemplateMultiLabelMetric(BaseMetric):
    """
    Template multi-label metric for multi-label classification.
    
    Users specify:
    - Input type (probabilities)
    - Threshold for binary conversion
    - Metric type (accuracy, f1, precision, recall)
    """
    
    def __init__(self, metric_type: str = "accuracy", threshold: float = 0.5):
        self.metric_type = metric_type
        self.threshold = threshold
        self.name = f"TemplateMultiLabel{metric_type.title()}"
        
    @property
    def expected_input_type(self) -> OutputType:
        """Expect probabilities for multi-label classification."""
        return OutputType.PROBABILITIES
    
    def get_name(self) -> str:
        """Return metric name."""
        return self.name
    
    def validate_inputs(self, predictions: List[Any], targets: List[Any]) -> bool:
        """Validate that inputs are compatible with multi-label classification."""
        if not predictions or not targets:
            return False
        
        # Check if predictions are lists/arrays of probabilities
        if not isinstance(predictions[0], (list, np.ndarray)):
            return False
        
        # Check if targets are dictionaries
        if not isinstance(targets[0], dict):
            return False
        
        return True
    
    def calculate(self, predictions: List[Any], targets: List[Any]) -> Dict[str, Any]:
        """
        Calculate multi-label metrics based on configured type.
        """
        if not self.validate_inputs(predictions, targets):
            return {"error": "Invalid inputs for multi-label classification"}
        
        # Convert predictions to binary using threshold
        binary_predictions = []
        binary_targets = []
        
        for pred, target in zip(predictions, targets):
            # Convert prediction to binary
            if isinstance(pred, list):
                pred_array = np.array(pred)
            else:
                pred_array = pred
            
            binary_pred = (pred_array > self.threshold).astype(int)
            binary_predictions.append(binary_pred)
            
            # Convert target to binary array
            target_array = []
            for emotion in ['anger', 'anticipation', 'disgust', 'fear', 'joy', 
                          'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']:
                # Handle both string and integer target values
                target_value = target.get(emotion, 0)
                if isinstance(target_value, str):
                    target_value = int(target_value)
                target_array.append(target_value)
            
            binary_targets.append(np.array(target_array))
        
        # Stack all predictions and targets
        all_predictions = np.vstack(binary_predictions)
        all_targets = np.vstack(binary_targets)
        
        # Calculate based on metric type
        if self.metric_type == "accuracy":
            return self._calculate_accuracy(all_predictions, all_targets)
        elif self.metric_type == "f1":
            return self._calculate_f1(all_predictions, all_targets)
        elif self.metric_type == "precision":
            return self._calculate_precision(all_predictions, all_targets)
        elif self.metric_type == "recall":
            return self._calculate_recall(all_predictions, all_targets)
        else:
            return {"error": f"Unknown metric type: {self.metric_type}"}
    
    def _calculate_accuracy(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        """Calculate per-emotion accuracy (not exact match)."""
        # Calculate accuracy per emotion, then average
        total_correct = 0
        total_predictions = 0
        
        for i in range(predictions.shape[1]):  # For each emotion
            emotion_correct = np.sum(predictions[:, i] == targets[:, i])
            total_correct += emotion_correct
            total_predictions += predictions.shape[0]
        
        accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0
        
        return {
            "accuracy": float(accuracy),
            "total_correct": int(total_correct),
            "total_predictions": int(total_predictions)
        }
    
    def _calculate_f1(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        """Calculate F1 score."""
        tp = np.sum((predictions == 1) & (targets == 1))
        fp = np.sum((predictions == 1) & (targets == 0))
        fn = np.sum((predictions == 0) & (targets == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall)
        }
    
    def _calculate_precision(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        """Calculate precision."""
        tp = np.sum((predictions == 1) & (targets == 1))
        fp = np.sum((predictions == 1) & (targets == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        return {"precision": float(precision)}
    
    def _calculate_recall(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        """Calculate recall."""
        tp = np.sum((predictions == 1) & (targets == 1))
        fn = np.sum((predictions == 0) & (targets == 1))
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return {"recall": float(recall)} 