"""
Accuracy metric for evaluating HuggingFace models in the BenchmarkEngine framework.

This metric calculates classification accuracy, precision, recall, and F1 score
for text classification tasks.
"""

from typing import Any, Dict, List

import numpy as np

from core.interfaces import BaseMetric, OutputType


class AccuracyMetric(BaseMetric):
    """
    Accuracy metric for text classification tasks.

    This metric calculates:
    - Accuracy: Percentage of correct predictions
    - Precision: True positives / (True positives + False positives)
    - Recall: True positives / (True positives + False negatives)
    - F1 Score: Harmonic mean of precision and recall

    Works with HuggingFace model outputs that contain:
    - "class": predicted class label
    - "prediction": predicted class label
    - "probabilities": probability distribution
    """

    def __init__(self):
        self.metric_name = "Accuracy"

    @property
    def expected_input_type(self) -> OutputType:
        """Accuracy metric expects class IDs."""
        return OutputType.CLASS_ID

    def calculate(
        self, predictions: List[Any], targets: List[Any], **kwargs
    ) -> Dict[str, float]:
        """
        Calculate accuracy metrics for classification predictions.

        Args:
            predictions: List of model predictions (dicts with "class" or "prediction" keys)
            targets: List of ground truth targets (integers)
            **kwargs: Additional arguments

        Returns:
            Dictionary containing accuracy metrics
        """
        if not self.validate_inputs(predictions, targets):
            return {"error": "Invalid inputs for accuracy calculation"}

        # Extract predicted classes from predictions
        predicted_classes = []
        for pred in predictions:
            if isinstance(pred, dict):
                # Handle different prediction formats
                if "class" in pred:
                    predicted_classes.append(pred["class"])
                elif "prediction" in pred:
                    predicted_classes.append(pred["prediction"])
                elif "probabilities" in pred:
                    # Get class with highest probability
                    probs = pred["probabilities"]
                    if isinstance(probs, list):
                        predicted_classes.append(np.argmax(probs))
                    else:
                        predicted_classes.append(0)  # Fallback
                else:
                    # Try to find any numeric value
                    numeric_values = [
                        v for v in pred.values() if isinstance(v, (int, float))
                    ]
                    if numeric_values:
                        predicted_classes.append(int(numeric_values[0]))
                    else:
                        predicted_classes.append(0)  # Fallback
            elif isinstance(pred, (int, float)):
                predicted_classes.append(int(pred))
            else:
                predicted_classes.append(0)  # Fallback

        # Convert to numpy arrays for calculations
        y_pred = np.array(predicted_classes)
        y_true = np.array(targets)

        # Calculate basic accuracy
        accuracy = np.mean(y_pred == y_true)

        # Calculate precision, recall, F1 for each class
        num_classes = max(len(np.unique(y_true)), len(np.unique(y_pred)), 2)

        # Handle binary classification
        if num_classes == 2:
            # For binary classification, calculate metrics for positive class (1)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            return {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1_score, 4),
                "num_samples": len(predictions),
                "num_classes": num_classes,
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
            }

        else:
            # For multi-class classification, calculate macro-averaged metrics
            from sklearn.metrics import precision_recall_fscore_support

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )

            return {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "num_samples": len(predictions),
                "num_classes": num_classes,
            }

    def get_name(self) -> str:
        """
        Return the name of this metric.

        Returns:
            Metric name
        """
        return self.metric_name

    def validate_inputs(self, predictions: List[Any], targets: List[Any]) -> bool:
        """
        Validate that inputs are compatible with this metric.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            True if inputs are valid for this metric
        """
        # Basic validation
        if not predictions or not targets:
            return False

        if len(predictions) != len(targets):
            return False

        # Check that targets are numeric
        try:
            targets_numeric = [int(t) for t in targets]
        except (ValueError, TypeError):
            return False

        # Check that predictions can be converted to classes
        try:
            for pred in predictions:
                if isinstance(pred, dict):
                    # Should have some way to extract class
                    if not any(
                        key in pred for key in ["class", "prediction", "probabilities"]
                    ):
                        return False
                elif not isinstance(pred, (int, float)):
                    return False
        except Exception:
            return False

        return True
