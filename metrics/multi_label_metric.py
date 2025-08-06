"""
Multi-Label Classification Metrics

This module provides metrics for multi-label classification tasks,
such as emotion detection with multiple emotion categories.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from core import BaseMetric, OutputType


class MultiLabelAccuracyMetric(BaseMetric):
    """
    Multi-label accuracy metric for emotion detection.
    
    Handles:
    - Multiple emotion categories (e.g., 11 emotions)
    - Binary classification per emotion
    - Threshold-based predictions
    - Micro and macro averaging
    """
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.name = "MultiLabelAccuracy"
        
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
        
        # Check if targets are dictionaries with emotion keys
        if not isinstance(targets[0], dict):
            return False
        
        return True
    
    def calculate(self, predictions: List[Any], targets: List[Any]) -> Dict[str, Any]:
        """
        Calculate multi-label accuracy metrics.
        
        Args:
            predictions: List of probability arrays [batch_size, num_emotions]
            targets: List of emotion dictionaries [batch_size, emotions]
            
        Returns:
            Dictionary with accuracy, precision, recall, f1_score
        """
        if not self.validate_inputs(predictions, targets):
            return {"error": "Invalid inputs for multi-label classification"}
        
        # Convert predictions to binary using threshold
        binary_predictions = []
        binary_targets = []
        
        for pred, target in zip(predictions, targets):
            # Convert prediction to binary (threshold-based)
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
                target_array.append(target.get(emotion, 0))
            
            binary_targets.append(np.array(target_array))
        
        # Stack all predictions and targets
        all_predictions = np.vstack(binary_predictions)  # [N, 11]
        all_targets = np.vstack(binary_targets)  # [N, 11]
        
        # Calculate metrics
        metrics = {}
        
        # Overall accuracy (exact match)
        exact_matches = np.all(all_predictions == all_targets, axis=1)
        accuracy = np.mean(exact_matches)
        metrics["accuracy"] = float(accuracy)
        
        # Per-emotion metrics
        emotion_names = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 
                        'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
        
        emotion_metrics = {}
        for i, emotion in enumerate(emotion_names):
            pred_emotion = all_predictions[:, i]
            target_emotion = all_targets[:, i]
            
            # Precision, Recall, F1 for this emotion
            tp = np.sum((pred_emotion == 1) & (target_emotion == 1))
            fp = np.sum((pred_emotion == 1) & (target_emotion == 0))
            fn = np.sum((pred_emotion == 0) & (target_emotion == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            emotion_metrics[emotion] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1)
            }
        
        metrics["emotion_metrics"] = emotion_metrics
        
        # Micro-averaged metrics (across all emotions)
        tp_total = np.sum((all_predictions == 1) & (all_targets == 1))
        fp_total = np.sum((all_predictions == 1) & (all_targets == 0))
        fn_total = np.sum((all_predictions == 0) & (all_targets == 1))
        
        micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        micro_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
        
        metrics["micro_precision"] = float(micro_precision)
        metrics["micro_recall"] = float(micro_recall)
        metrics["micro_f1"] = float(micro_f1)
        
        # Macro-averaged metrics (average across emotions)
        macro_precision = np.mean([emotion_metrics[emotion]["precision"] for emotion in emotion_names])
        macro_recall = np.mean([emotion_metrics[emotion]["recall"] for emotion in emotion_names])
        macro_f1 = np.mean([emotion_metrics[emotion]["f1_score"] for emotion in emotion_names])
        
        metrics["macro_precision"] = float(macro_precision)
        metrics["macro_recall"] = float(macro_recall)
        metrics["macro_f1"] = float(macro_f1)
        
        return metrics


class MultiLabelF1Metric(BaseMetric):
    """
    F1-score metric for multi-label classification.
    """
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.name = "MultiLabelF1"
        
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
        
        # Check if targets are dictionaries with emotion keys
        if not isinstance(targets[0], dict):
            return False
        
        return True
    
    def calculate(self, predictions: List[Any], targets: List[Any]) -> Dict[str, Any]:
        """
        Calculate F1-score for multi-label classification.
        """
        if not self.validate_inputs(predictions, targets):
            return {"error": "Invalid inputs for multi-label classification"}
        
        # Use the same calculation as MultiLabelAccuracyMetric
        metric = MultiLabelAccuracyMetric(self.threshold)
        results = metric.calculate(predictions, targets)
        
        # Return only F1-related metrics
        return {
            "micro_f1": results.get("micro_f1", 0.0),
            "macro_f1": results.get("macro_f1", 0.0),
            "emotion_f1_scores": {
                emotion: metrics["f1_score"] 
                for emotion, metrics in results.get("emotion_metrics", {}).items()
            }
        } 