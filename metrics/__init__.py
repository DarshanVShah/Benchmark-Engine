"""
Metrics implementations for the BenchmarkEngine framework.

This package contains metric implementations for evaluating model performance:
- Task-specific metrics (classification, regression, emotion detection, sentiment analysis)
- Configurable metrics with custom evaluation criteria
- Generic metrics for auto-detection
- Custom metrics (fairness, robustness, etc.)
"""

from .dummy_metric import DummyAccuracyMetric
from .accuracy_metric import AccuracyMetric
from .generic_metric import GenericMetric
from .task_metric import (
    ClassificationMetric,
    RegressionMetric,
    ConfigurableMetric
)

__all__ = [
    'DummyAccuracyMetric',
    'AccuracyMetric',
    'GenericMetric',
    'ClassificationMetric',
    'RegressionMetric',
    'ConfigurableMetric'
] 