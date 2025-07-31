"""
Generic Metric for Multiple Task Types

This metric automatically detects the task type and applies appropriate evaluation
metrics without requiring custom metric files for each dataset.
"""

import numpy as np
from typing import Dict, Any, List
from core import BaseMetric


class GenericMetric(BaseMetric):
    """
    Generic metric that automatically handles different task types.
    
    Supports:
    - Text classification (accuracy, precision, recall, F1)
    - Regression (MSE, MAE, correlation)
    - Emotion detection (VAD correlation, MSE)
    - Sentiment analysis (accuracy, correlation)
    
    Users don't need custom metric files - this handles everything!
    """
    
    def __init__(self): 
        self.name = "GenericMetric"
        self.task_type = "auto-detected"
        
    def calculate(self, predictions: List[Any], targets: List[Any], **kwargs) -> Dict[str, float]:
        """Calculate metrics based on automatically detected task type."""
        if not predictions or not targets:
            return {"error": "No predictions or targets provided"}
        
        # Auto-detect task type from first target
        self._detect_task_type(targets[0])
        
        # Calculate appropriate metrics
        if self.task_type == "classification":
            return self._calculate_classification_metrics(predictions, targets)
        elif self.task_type == "emotion-detection":
            return self._calculate_emotion_metrics(predictions, targets)
        elif self.task_type == "regression":
            return self._calculate_regression_metrics(predictions, targets)
        else:
            return self._calculate_generic_metrics(predictions, targets)
    
    def _detect_task_type(self, sample_target: Any):
        """Detect task type from target format."""
        if isinstance(sample_target, dict):
            if "valence" in sample_target or "arousal" in sample_target:
                self.task_type = "emotion-detection"
            else:
                self.task_type = "classification"
        elif isinstance(sample_target, (int, float)):
            if isinstance(sample_target, int):
                self.task_type = "classification"
            else:
                self.task_type = "regression"
        else:
            self.task_type = "classification"
    
    def _calculate_classification_metrics(self, predictions: List[Any], targets: List[Any]) -> Dict[str, float]:
        """Calculate classification metrics."""
        # Extract predicted classes
        pred_classes = []
        for pred in predictions:
            if isinstance(pred, dict):
                if "class" in pred:
                    pred_classes.append(pred["class"])
                elif "prediction" in pred:
                    pred_classes.append(pred["prediction"])
                else:
                    # Use highest probability class
                    probs = pred.get("probabilities", [])
                    if probs:
                        pred_classes.append(np.argmax(probs))
                    else:
                        pred_classes.append(0)
            else:
                pred_classes.append(int(pred))
        
        # Extract target classes
        target_classes = []
        for target in targets:
            if isinstance(target, dict):
                target_classes.append(target.get("label", 0))
            else:
                target_classes.append(int(target))
        
        # Calculate metrics
        correct = sum(1 for p, t in zip(pred_classes, target_classes) if p == t)
        accuracy = correct / len(pred_classes) if pred_classes else 0
        
        # Calculate precision, recall, F1 for binary classification
        if len(set(target_classes)) == 2:  # Binary classification
            tp = sum(1 for p, t in zip(pred_classes, target_classes) if p == 1 and t == 1)
            fp = sum(1 for p, t in zip(pred_classes, target_classes) if p == 1 and t == 0)
            fn = sum(1 for p, t in zip(pred_classes, target_classes) if p == 0 and t == 1)
            tn = sum(1 for p, t in zip(pred_classes, target_classes) if p == 0 and t == 0)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
                "num_samples": len(predictions)
            }
        else:
            return {
                "accuracy": accuracy,
                "num_samples": len(predictions)
            }
    
    def _calculate_emotion_metrics(self, predictions: List[Any], targets: List[Any]) -> Dict[str, float]:
        """Calculate emotion detection (VAD) metrics."""
        pred_vad = []
        target_vad = []
        
        for pred, target in zip(predictions, targets):
            # Extract VAD from predictions
            if isinstance(pred, dict):
                pred_v = pred.get("valence", 0.0)
                pred_a = pred.get("arousal", 0.0)
                pred_d = pred.get("dominance", 0.0)
            else:
                # Assume single value or list
                if isinstance(pred, (list, tuple)) and len(pred) >= 3:
                    pred_v, pred_a, pred_d = pred[0], pred[1], pred[2]
                else:
                    pred_v = pred_a = pred_d = float(pred) if pred is not None else 0.0
            
            # Extract VAD from targets
            if isinstance(target, dict):
                target_v = target.get("valence", 0.0)
                target_a = target.get("arousal", 0.0)
                target_d = target.get("dominance", 0.0)
            else:
                if isinstance(target, (list, tuple)) and len(target) >= 3:
                    target_v, target_a, target_d = target[0], target[1], target[2]
                else:
                    target_v = target_a = target_d = float(target) if target is not None else 0.0
            
            pred_vad.append([pred_v, pred_a, pred_d])
            target_vad.append([target_v, target_a, target_d])
        
        # Convert to numpy arrays
        pred_array = np.array(pred_vad)
        target_array = np.array(target_vad)
        
        # Calculate metrics for each VAD dimension
        metrics = {}
        dimensions = ["valence", "arousal", "dominance"]
        
        for i, dim in enumerate(dimensions):
            pred_dim = pred_array[:, i]
            target_dim = target_array[:, i]
            
            # Correlation
            correlation = np.corrcoef(pred_dim, target_dim)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            
            # MSE and MAE
            mse = np.mean((pred_dim - target_dim) ** 2)
            mae = np.mean(np.abs(pred_dim - target_dim))
            
            metrics[f"{dim}_correlation"] = float(correlation)
            metrics[f"{dim}_mse"] = float(mse)
            metrics[f"{dim}_mae"] = float(mae)
        
        # Overall metrics
        overall_mse = np.mean((pred_array - target_array) ** 2)
        overall_mae = np.mean(np.abs(pred_array - target_array))
        avg_correlation = np.mean([metrics[f"{dim}_correlation"] for dim in dimensions])
        
        metrics.update({
            "overall_mse": float(overall_mse),
            "overall_mae": float(overall_mae),
            "avg_correlation": float(avg_correlation),
            "num_samples": len(predictions)
        })
        
        return metrics
    
    def _calculate_regression_metrics(self, predictions: List[Any], targets: List[Any]) -> Dict[str, float]:
        """Calculate regression metrics."""
        # Extract numeric values
        pred_values = []
        target_values = []
        
        for pred, target in zip(predictions, targets):
            if isinstance(pred, dict):
                pred_val = pred.get("prediction", pred.get("value", 0.0))
            else:
                pred_val = float(pred) if pred is not None else 0.0
            
            if isinstance(target, dict):
                target_val = target.get("value", target.get("target", 0.0))
            else:
                target_val = float(target) if target is not None else 0.0
            
            pred_values.append(pred_val)
            target_values.append(target_val)
        
        # Calculate metrics
        pred_array = np.array(pred_values)
        target_array = np.array(target_values)
        
        mse = np.mean((pred_array - target_array) ** 2)
        mae = np.mean(np.abs(pred_array - target_array))
        rmse = np.sqrt(mse)
        
        # Correlation
        correlation = np.corrcoef(pred_array, target_array)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        return {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "correlation": float(correlation),
            "num_samples": len(predictions)
        }
    
    def _calculate_generic_metrics(self, predictions: List[Any], targets: List[Any]) -> Dict[str, float]:
        """Calculate generic metrics for unknown task types."""
        return {
            "num_samples": len(predictions),
            "task_type": self.task_type,
            "note": "Generic metrics calculated"
        }
    
    def get_name(self) -> str:
        """Return the name of this metric."""
        return self.name
    
    def validate_inputs(self, predictions: List[Any], targets: List[Any]) -> bool:
        """Validate that inputs are compatible."""
        if not predictions or not targets:
            return False
        
        if len(predictions) != len(targets):
            return False
        
        # Basic validation - accept any format
        return True