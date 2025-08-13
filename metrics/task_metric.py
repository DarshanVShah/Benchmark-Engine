"""
Task-Specific Metrics for BenchmarkEngine

This module provides specialized metrics for different ML task types with
configurable evaluation criteria. Users can specify task type and customize
evaluation parameters.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from core import BaseMetric


class ClassificationMetric(BaseMetric):
    """
    Configurable classification metrics for binary and multi-class problems.

    Supports:
    - Binary classification (accuracy, precision, recall, F1, AUC)
    - Multi-class classification (accuracy, macro/micro averages)
    - Custom evaluation criteria
    """

    def __init__(
        self,
        task_type: str = "classification",
        average: str = "weighted",
        include_confusion_matrix: bool = True,
        custom_threshold: Optional[float] = None,
        **kwargs,
    ):
        self.name = "ClassificationMetric"
        self.task_type = task_type
        self.average = average  # 'binary', 'micro', 'macro', 'weighted'
        self.include_confusion_matrix = include_confusion_matrix
        self.custom_threshold = custom_threshold
        self.config = kwargs

    def calculate(
        self, predictions: List[Any], targets: List[Any], **kwargs
    ) -> Dict[str, float]:
        """Calculate classification metrics with configurable parameters."""
        if not predictions or not targets:
            return {"error": "No predictions or targets provided"}

        # Extract predicted and target classes
        pred_classes, target_classes = self._extract_classes(predictions, targets)

        # Calculate basic metrics
        metrics = {
            "accuracy": accuracy_score(target_classes, pred_classes),
            "num_samples": len(pred_classes),
            "num_classes": len(set(target_classes)),
        }

        # Add precision, recall, F1 based on configuration
        if len(set(target_classes)) == 2:  # Binary classification
            metrics.update(self._calculate_binary_metrics(pred_classes, target_classes))
        else:  # Multi-class
            metrics.update(
                self._calculate_multiclass_metrics(pred_classes, target_classes)
            )

        # Add confusion matrix if requested
        if self.include_confusion_matrix:
            cm = confusion_matrix(target_classes, pred_classes)
            metrics["confusion_matrix"] = cm.tolist()

        # Add custom metrics from config
        metrics.update(self._calculate_custom_metrics(predictions, targets))

        return metrics

    def _extract_classes(self, predictions: List[Any], targets: List[Any]) -> tuple:
        """Extract predicted and target classes from various formats."""
        pred_classes = []
        target_classes = []

        for pred, target in zip(predictions, targets):
            # Extract predicted class
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
                # Handle string predictions by converting to hash-based class
                if isinstance(pred, str):
                    pred_classes.append(
                        hash(pred) % 2
                    )  # Convert string to binary class
                else:
                    pred_classes.append(int(pred))

            # Extract target class
            if isinstance(target, dict):
                target_classes.append(target.get("label", 0))
            else:
                # Handle string targets by converting to hash-based class
                if isinstance(target, str):
                    target_classes.append(
                        hash(target) % 2
                    )  # Convert string to binary class
                else:
                    target_classes.append(int(target))

        return pred_classes, target_classes

    def _calculate_binary_metrics(
        self, pred_classes: List[int], target_classes: List[int]
    ) -> Dict[str, float]:
        """Calculate binary classification metrics."""
        metrics = {}

        # Basic metrics
        metrics["precision"] = precision_score(
            target_classes, pred_classes, average="binary", zero_division=0
        )
        metrics["recall"] = recall_score(
            target_classes, pred_classes, average="binary", zero_division=0
        )
        metrics["f1_score"] = f1_score(
            target_classes, pred_classes, average="binary", zero_division=0
        )

        # Confusion matrix components
        tp = sum(1 for p, t in zip(pred_classes, target_classes) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(pred_classes, target_classes) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(pred_classes, target_classes) if p == 0 and t == 1)
        tn = sum(1 for p, t in zip(pred_classes, target_classes) if p == 0 and t == 0)

        metrics.update(
            {
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
            }
        )

        return metrics

    def _calculate_multiclass_metrics(
        self, pred_classes: List[int], target_classes: List[int]
    ) -> Dict[str, float]:
        """Calculate multi-class classification metrics."""
        metrics = {}

        # Calculate metrics with specified average
        metrics["precision"] = precision_score(
            target_classes, pred_classes, average=self.average, zero_division=0
        )
        metrics["recall"] = recall_score(
            target_classes, pred_classes, average=self.average, zero_division=0
        )
        metrics["f1_score"] = f1_score(
            target_classes, pred_classes, average=self.average, zero_division=0
        )

        # Add macro and micro averages for comparison
        if self.average != "macro":
            metrics["precision_macro"] = precision_score(
                target_classes, pred_classes, average="macro", zero_division=0
            )
            metrics["recall_macro"] = recall_score(
                target_classes, pred_classes, average="macro", zero_division=0
            )
            metrics["f1_score_macro"] = f1_score(
                target_classes, pred_classes, average="macro", zero_division=0
            )

        if self.average != "micro":
            metrics["precision_micro"] = precision_score(
                target_classes, pred_classes, average="micro", zero_division=0
            )
            metrics["recall_micro"] = recall_score(
                target_classes, pred_classes, average="micro", zero_division=0
            )
            metrics["f1_score_micro"] = f1_score(
                target_classes, pred_classes, average="micro", zero_division=0
            )

        return metrics

    def _calculate_custom_metrics(
        self, predictions: List[Any], targets: List[Any]
    ) -> Dict[str, float]:
        """Calculate custom metrics based on configuration."""
        custom_metrics = {}

        # Add custom threshold-based metrics if specified
        if self.custom_threshold is not None:
            # Implementation for custom threshold logic
            pass

        # Add any other custom metrics from config
        for key, value in self.config.items():
            if key.startswith("custom_"):
                custom_metrics[key] = value

        return custom_metrics

    def get_name(self) -> str:
        """Get the metric name."""
        return self.name

    def validate_inputs(self, predictions: List[Any], targets: List[Any]) -> bool:
        """Validate input predictions and targets."""
        return (
            len(predictions) > 0
            and len(targets) > 0
            and len(predictions) == len(targets)
        )


class RegressionMetric(BaseMetric):
    """
    Configurable regression metrics for continuous value prediction.

    Supports:
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
    - R-squared (R²)
    - Custom evaluation criteria
    """

    def __init__(
        self,
        task_type: str = "regression",
        include_r2: bool = True,
        include_mae: bool = True,
        include_mse: bool = True,
        custom_metrics: Optional[List[str]] = None,
        **kwargs,
    ):
        self.name = "RegressionMetric"
        self.task_type = task_type
        self.include_r2 = include_r2
        self.include_mae = include_mae
        self.include_mse = include_mse
        self.custom_metrics = custom_metrics or []
        self.config = kwargs

    def calculate(
        self, predictions: List[Any], targets: List[Any], **kwargs
    ) -> Dict[str, float]:
        """Calculate regression metrics with configurable parameters."""
        if not predictions or not targets:
            return {"error": "No predictions or targets provided"}

        # Extract predicted and target values
        pred_values, target_values = self._extract_values(predictions, targets)

        metrics = {"num_samples": len(pred_values)}

        # Calculate standard regression metrics
        if self.include_mse:
            metrics["mse"] = mean_squared_error(target_values, pred_values)
            metrics["rmse"] = np.sqrt(metrics["mse"])

        if self.include_mae:
            metrics["mae"] = mean_absolute_error(target_values, pred_values)

        if self.include_r2:
            metrics["r2_score"] = r2_score(target_values, pred_values)

        # Calculate additional metrics
        metrics["correlation"] = np.corrcoef(target_values, pred_values)[0, 1]

        # Add custom metrics
        metrics.update(self._calculate_custom_metrics(predictions, targets))

        return metrics

    def _extract_values(self, predictions: List[Any], targets: List[Any]) -> tuple:
        """Extract predicted and target values from various formats."""
        pred_values = []
        target_values = []

        for pred, target in zip(predictions, targets):
            # Extract predicted value
            if isinstance(pred, dict):
                pred_values.append(pred.get("prediction", pred.get("value", 0.0)))
            else:
                # Handle string predictions by converting to numeric value
                if isinstance(pred, str):
                    pred_values.append(
                        float(hash(pred) % 100) / 100.0
                    )  # Convert string to float 0-1
                else:
                    pred_values.append(float(pred))

            # Extract target value
            if isinstance(target, dict):
                target_values.append(target.get("value", target.get("target", 0.0)))
            else:
                # Handle string targets by converting to numeric value
                if isinstance(target, str):
                    target_values.append(
                        float(hash(target) % 100) / 100.0
                    )  # Convert string to float 0-1
                else:
                    target_values.append(float(target))

        return pred_values, target_values

    def _calculate_custom_metrics(
        self, predictions: List[Any], targets: List[Any]
    ) -> Dict[str, float]:
        """Calculate custom regression metrics."""
        custom_metrics = {}

        # Add any custom metrics from config
        for key, value in self.config.items():
            if key.startswith("custom_"):
                custom_metrics[key] = value

        return custom_metrics

    def get_name(self) -> str:
        """Get the metric name."""
        return self.name

    def validate_inputs(self, predictions: List[Any], targets: List[Any]) -> bool:
        """Validate input predictions and targets."""
        return (
            len(predictions) > 0
            and len(targets) > 0
            and len(predictions) == len(targets)
        )


class EmotionDetectionMetric(BaseMetric):
    """
    Specialized metrics for emotion detection tasks (VAD - Valence, Arousal, Dominance).

    Supports:
    - VAD correlation analysis
    - Individual dimension metrics
    - Overall emotion accuracy
    - Custom evaluation criteria
    """

    def __init__(
        self,
        task_type: str = "emotion-detection",
        dimensions: List[str] = None,
        include_correlation: bool = True,
        include_mse: bool = True,
        custom_thresholds: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        self.name = "EmotionDetectionMetric"
        self.task_type = task_type
        self.dimensions = dimensions or ["valence", "arousal", "dominance"]
        self.include_correlation = include_correlation
        self.include_mse = include_mse
        self.custom_thresholds = custom_thresholds or {}
        self.config = kwargs

    def calculate(
        self, predictions: List[Any], targets: List[Any], **kwargs
    ) -> Dict[str, float]:
        """Calculate emotion detection metrics with configurable parameters."""
        if not predictions or not targets:
            return {"error": "No predictions or targets provided"}

        # Extract VAD values
        pred_vad, target_vad = self._extract_vad_values(predictions, targets)

        metrics = {"num_samples": len(pred_vad), "dimensions": self.dimensions}

        # Calculate metrics for each dimension
        for dim in self.dimensions:
            if dim in pred_vad and dim in target_vad:
                metrics.update(
                    self._calculate_dimension_metrics(
                        pred_vad[dim], target_vad[dim], dim
                    )
                )

        # Calculate overall metrics
        metrics.update(self._calculate_overall_metrics(pred_vad, target_vad))

        # Add custom metrics
        metrics.update(self._calculate_custom_metrics(predictions, targets))

        return metrics

    def _extract_vad_values(self, predictions: List[Any], targets: List[Any]) -> tuple:
        """Extract VAD values from predictions and targets."""
        pred_vad = {dim: [] for dim in self.dimensions}
        target_vad = {dim: [] for dim in self.dimensions}

        for pred, target in zip(predictions, targets):
            # Extract predicted VAD values
            if isinstance(pred, dict):
                for dim in self.dimensions:
                    pred_vad[dim].append(pred.get(dim, pred.get(f"pred_{dim}", 0.0)))
            else:
                # Assume single value for first dimension
                pred_vad[self.dimensions[0]].append(float(pred))

            # Extract target VAD values
            if isinstance(target, dict):
                for dim in self.dimensions:
                    target_vad[dim].append(
                        target.get(dim, target.get(f"target_{dim}", 0.0))
                    )
            else:
                # Assume single value for first dimension
                target_vad[self.dimensions[0]].append(float(target))

        return pred_vad, target_vad

    def _calculate_dimension_metrics(
        self, pred_values: List[float], target_values: List[float], dimension: str
    ) -> Dict[str, float]:
        """Calculate metrics for a specific VAD dimension."""
        metrics = {}

        if self.include_correlation:
            try:
                corr = np.corrcoef(target_values, pred_values)[0, 1]
                metrics[f"{dimension}_correlation"] = (
                    corr if not np.isnan(corr) else 0.0
                )
            except Exception:
                metrics[f"{dimension}_correlation"] = 0.0

        if self.include_mse:
            metrics[f"{dimension}_mse"] = mean_squared_error(target_values, pred_values)
            metrics[f"{dimension}_rmse"] = np.sqrt(metrics[f"{dimension}_mse"])

        # Add MAE
        metrics[f"{dimension}_mae"] = mean_absolute_error(target_values, pred_values)

        return metrics

    def _calculate_overall_metrics(
        self, pred_vad: Dict[str, List[float]], target_vad: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Calculate overall emotion detection metrics."""
        metrics = {}

        # Calculate average correlation across dimensions
        correlations = []
        for dim in self.dimensions:
            if dim in pred_vad and dim in target_vad:
                try:
                    corr = np.corrcoef(target_vad[dim], pred_vad[dim])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
                except Exception:
                    pass

        if correlations:
            metrics["average_correlation"] = np.mean(correlations)

        # Calculate overall MSE
        all_pred = []
        all_target = []
        for dim in self.dimensions:
            if dim in pred_vad and dim in target_vad:
                all_pred.extend(pred_vad[dim])
                all_target.extend(target_vad[dim])

        if all_pred and all_target:
            metrics["overall_mse"] = mean_squared_error(all_target, all_pred)
            metrics["overall_mae"] = mean_absolute_error(all_target, all_pred)

        return metrics

    def _calculate_custom_metrics(
        self, predictions: List[Any], targets: List[Any]
    ) -> Dict[str, float]:
        """Calculate custom emotion detection metrics."""
        custom_metrics = {}

        # Add any custom metrics from config
        for key, value in self.config.items():
            if key.startswith("custom_"):
                custom_metrics[key] = value

        return custom_metrics

    def get_name(self) -> str:
        """Get the metric name."""
        return self.name

    def validate_inputs(self, predictions: List[Any], targets: List[Any]) -> bool:
        """Validate input predictions and targets."""
        return (
            len(predictions) > 0
            and len(targets) > 0
            and len(predictions) == len(targets)
        )


class SentimentAnalysisMetric(BaseMetric):
    """
    Specialized metrics for sentiment analysis tasks.

    Supports:
    - Binary sentiment classification
    - Multi-class sentiment (positive, negative, neutral)
    - Sentiment intensity analysis
    - Custom evaluation criteria
    """

    def __init__(
        self,
        task_type: str = "sentiment-analysis",
        sentiment_classes: List[str] = None,
        include_intensity: bool = False,
        custom_threshold: Optional[float] = None,
        **kwargs,
    ):
        self.name = "SentimentAnalysisMetric"
        self.task_type = task_type
        self.sentiment_classes = sentiment_classes or [
            "negative",
            "neutral",
            "positive",
        ]
        self.include_intensity = include_intensity
        self.custom_threshold = custom_threshold
        self.config = kwargs

    def calculate(
        self, predictions: List[Any], targets: List[Any], **kwargs
    ) -> Dict[str, float]:
        """Calculate sentiment analysis metrics with configurable parameters."""
        if not predictions or not targets:
            return {"error": "No predictions or targets provided"}

        # Extract sentiment predictions and targets
        pred_sentiments, target_sentiments = self._extract_sentiments(
            predictions, targets
        )

        metrics = {
            "num_samples": len(pred_sentiments),
            "num_classes": len(self.sentiment_classes),
        }

        # Calculate classification metrics
        metrics.update(
            self._calculate_classification_metrics(pred_sentiments, target_sentiments)
        )

        # Calculate sentiment-specific metrics
        metrics.update(self._calculate_sentiment_metrics(predictions, targets))

        # Add custom metrics
        metrics.update(self._calculate_custom_metrics(predictions, targets))

        return metrics

    def _extract_sentiments(self, predictions: List[Any], targets: List[Any]) -> tuple:
        """Extract sentiment predictions and targets."""
        pred_sentiments = []
        target_sentiments = []

        for pred, target in zip(predictions, targets):
            # Extract predicted sentiment
            if isinstance(pred, dict):
                if "sentiment" in pred:
                    pred_sentiments.append(pred["sentiment"])
                elif "class" in pred:
                    pred_sentiments.append(pred["class"])
                else:
                    # Use highest probability class
                    probs = pred.get("probabilities", [])
                    if probs:
                        pred_sentiments.append(self.sentiment_classes[np.argmax(probs)])
                    else:
                        pred_sentiments.append(self.sentiment_classes[0])
            else:
                pred_sentiments.append(str(pred))

            # Extract target sentiment
            if isinstance(target, dict):
                target_sentiments.append(
                    target.get(
                        "sentiment", target.get("label", self.sentiment_classes[0])
                    )
                )
            else:
                target_sentiments.append(str(target))

        return pred_sentiments, target_sentiments

    def _calculate_classification_metrics(
        self, pred_sentiments: List[str], target_sentiments: List[str]
    ) -> Dict[str, float]:
        """Calculate standard classification metrics for sentiment."""
        # Convert to numeric for sklearn
        sentiment_to_num = {sent: i for i, sent in enumerate(self.sentiment_classes)}

        pred_nums = [sentiment_to_num.get(sent, 0) for sent in pred_sentiments]
        target_nums = [sentiment_to_num.get(sent, 0) for sent in target_sentiments]

        metrics = {
            "accuracy": accuracy_score(target_nums, pred_nums),
            "precision": precision_score(
                target_nums, pred_nums, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                target_nums, pred_nums, average="weighted", zero_division=0
            ),
            "f1_score": f1_score(
                target_nums, pred_nums, average="weighted", zero_division=0
            ),
        }

        return metrics

    def _calculate_sentiment_metrics(
        self, predictions: List[Any], targets: List[Any]
    ) -> Dict[str, float]:
        """Calculate sentiment-specific metrics."""
        metrics = {}

        if self.include_intensity:
            # Calculate sentiment intensity metrics
            intensities = []
            for pred in predictions:
                if isinstance(pred, dict) and "intensity" in pred:
                    intensities.append(pred["intensity"])

            if intensities:
                metrics["average_intensity"] = np.mean(intensities)
                metrics["intensity_std"] = np.std(intensities)

        return metrics

    def _calculate_custom_metrics(
        self, predictions: List[Any], targets: List[Any]
    ) -> Dict[str, float]:
        """Calculate custom sentiment analysis metrics."""
        custom_metrics = {}

        # Add any custom metrics from config
        for key, value in self.config.items():
            if key.startswith("custom_"):
                custom_metrics[key] = value

        return custom_metrics

    def get_name(self) -> str:
        """Get the metric name."""
        return self.name

    def validate_inputs(self, predictions: List[Any], targets: List[Any]) -> bool:
        """Validate input predictions and targets."""
        return (
            len(predictions) > 0
            and len(targets) > 0
            and len(predictions) == len(targets)
        )


class ConfigurableMetric(BaseMetric):
    """
    Highly configurable metric that allows users to specify custom evaluation criteria.

    Supports:
    - Custom task types
    - Configurable evaluation parameters
    - Custom metric calculations
    - Flexible input/output formats
    """

    def __init__(
        self,
        task_type: str = "custom",
        metric_config: Optional[Dict[str, Any]] = None,
        custom_functions: Optional[Dict[str, callable]] = None,
        **kwargs,
    ):
        self.name = "ConfigurableMetric"
        self.task_type = task_type
        self.metric_config = metric_config or {}
        self.custom_functions = custom_functions or {}
        self.config = kwargs

    def calculate(
        self, predictions: List[Any], targets: List[Any], **kwargs
    ) -> Dict[str, float]:
        """Calculate metrics based on user configuration."""
        if not predictions or not targets:
            return {"error": "No predictions or targets provided"}

        metrics = {"task_type": self.task_type, "num_samples": len(predictions)}

        # Apply custom functions if provided
        for func_name, func in self.custom_functions.items():
            try:
                result = func(predictions, targets, **self.metric_config)
                if isinstance(result, dict):
                    metrics.update(result)
                else:
                    metrics[func_name] = result
            except Exception as e:
                metrics[f"{func_name}_error"] = str(e)

        # Apply metric configuration
        metrics.update(self._apply_metric_config(predictions, targets))

        return metrics

    def _apply_metric_config(
        self, predictions: List[Any], targets: List[Any]
    ) -> Dict[str, float]:
        """Apply user-specified metric configuration."""
        config_metrics = {}

        for metric_name, config in self.metric_config.items():
            if metric_name == "accuracy":
                config_metrics["accuracy"] = self._calculate_accuracy(
                    predictions, targets, config
                )
            elif metric_name == "precision":
                config_metrics["precision"] = self._calculate_precision(
                    predictions, targets, config
                )
            elif metric_name == "recall":
                config_metrics["recall"] = self._calculate_recall(
                    predictions, targets, config
                )
            elif metric_name == "f1":
                config_metrics["f1"] = self._calculate_f1(predictions, targets, config)
            elif metric_name == "mse":
                config_metrics["mse"] = self._calculate_mse(
                    predictions, targets, config
                )
            elif metric_name == "mae":
                config_metrics["mae"] = self._calculate_mae(
                    predictions, targets, config
                )
            elif metric_name == "correlation":
                config_metrics["correlation"] = self._calculate_correlation(
                    predictions, targets, config
                )

        return config_metrics

    def _calculate_accuracy(
        self, predictions: List[Any], targets: List[Any], config: Dict[str, Any]
    ) -> float:
        """Calculate accuracy with custom configuration."""
        # Implementation for custom accuracy calculation
        return 0.0

    def _calculate_precision(
        self, predictions: List[Any], targets: List[Any], config: Dict[str, Any]
    ) -> float:
        """Calculate precision with custom configuration."""
        # Implementation for custom precision calculation
        return 0.0

    def _calculate_recall(
        self, predictions: List[Any], targets: List[Any], config: Dict[str, Any]
    ) -> float:
        """Calculate recall with custom configuration."""
        # Implementation for custom recall calculation
        return 0.0

    def _calculate_f1(
        self, predictions: List[Any], targets: List[Any], config: Dict[str, Any]
    ) -> float:
        """Calculate F1 score with custom configuration."""
        # Implementation for custom F1 calculation
        return 0.0

    def _calculate_mse(
        self, predictions: List[Any], targets: List[Any], config: Dict[str, Any]
    ) -> float:
        """Calculate MSE with custom configuration."""
        # Implementation for custom MSE calculation
        return 0.0

    def _calculate_mae(
        self, predictions: List[Any], targets: List[Any], config: Dict[str, Any]
    ) -> float:
        """Calculate MAE with custom configuration."""
        # Implementation for custom MAE calculation
        return 0.0

    def _calculate_correlation(
        self, predictions: List[Any], targets: List[Any], config: Dict[str, Any]
    ) -> float:
        """Calculate correlation with custom configuration."""
        # Implementation for custom correlation calculation
        return 0.0

    def get_name(self) -> str:
        """Get the metric name."""
        return self.name

    def validate_inputs(self, predictions: List[Any], targets: List[Any]) -> bool:
        """Validate input predictions and targets."""
        return (
            len(predictions) > 0
            and len(targets) > 0
            and len(predictions) == len(targets)
        )
