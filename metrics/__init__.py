"""
Metrics implementations for the BenchmarkEngine framework.

This package contains metric implementations for evaluating model performance:
- Accuracy metrics (classification, regression)
- Latency metrics (inference time, throughput)
- Memory metrics (peak memory, memory efficiency)
- Custom metrics (fairness, robustness, etc.)
"""

from .dummy_metric import DummyAccuracyMetric
from .accuracy_metric import AccuracyMetric

__all__ = [
    'DummyAccuracyMetric',
    'AccuracyMetric'
] 