"""
Configurable Metrics Example

This example demonstrates how to use task-specific metrics with custom configurations
for different ML tasks. Users can specify task type and customize evaluation criteria.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import BenchmarkEngine
from plugins import DummyModelAdapter
from benchmark_datasets import DummyDataset
from metrics import ClassificationMetric, RegressionMetric, ConfigurableMetric


def main():
    """Demonstrate configurable task-specific metrics."""

    print("Configurable Metrics Demo")

    # Create the benchmark engine
    engine = BenchmarkEngine()

    # Register basic components
    engine.register_adapter("dummy", DummyModelAdapter)
    engine.register_dataset("dummy", DummyDataset)

    # Load model and dataset
    engine.load_model("dummy", "path/to/dummy/model")
    engine.load_dataset("dummy", "path/to/dummy/dataset")

    # Demo 1: Classification with custom configuration
    print("\n1. Classification Metrics with Custom Configuration")

    # Create a custom classification metric class
    class CustomClassificationMetric(ClassificationMetric):
        def __init__(self):
            super().__init__(
                task_type="classification",
                average="weighted",  # 'binary', 'micro', 'macro', 'weighted'
                include_confusion_matrix=True,
                custom_threshold=0.5,
            )

    engine.register_metric("classification", CustomClassificationMetric)
    engine.add_metric("classification")

    results = engine.run_benchmark(num_samples=20, warmup_runs=2)
    print("Classification Results:")
    engine.print_results()

    # Demo 2: Regression with custom configuration
    print("\n2. Regression Metrics with Custom Configuration")

    # Create a custom regression metric class
    class CustomRegressionMetric(RegressionMetric):
        def __init__(self):
            super().__init__(
                task_type="regression",
                include_r2=True,
                include_mae=True,
                include_mse=True,
                custom_metrics=["custom_metric_1", "custom_metric_2"],
            )

    engine.register_metric("regression", CustomRegressionMetric)
    engine.add_metric("regression")

    results = engine.run_benchmark(num_samples=20, warmup_runs=2)
    print("Regression Results:")
    engine.print_results()

    # Demo 3: Highly configurable custom metric
    print("\n3. Configurable Custom Metric")

    # Define custom metric functions
    def custom_accuracy(predictions, targets, config):
        """Custom accuracy calculation."""
        correct = sum(1 for p, t in zip(predictions, targets) if p == t)
        return correct / len(predictions) if predictions else 0

    def custom_precision(predictions, targets, config):
        """Custom precision calculation."""
        threshold = config.get("threshold", 0.5)
        tp = sum(
            1 for p, t in zip(predictions, targets) if p >= threshold and t >= threshold
        )
        fp = sum(
            1 for p, t in zip(predictions, targets) if p >= threshold and t < threshold
        )
        return tp / (tp + fp) if (tp + fp) > 0 else 0

    # Create a custom configurable metric class
    class CustomConfigurableMetric(ConfigurableMetric):
        def __init__(self):
            super().__init__(
                task_type="custom",
                metric_config={
                    "accuracy": {"threshold": 0.5},
                    "precision": {"threshold": 0.7},
                    "custom_metric": {"param1": "value1"},
                },
                custom_functions={
                    "custom_accuracy": custom_accuracy,
                    "custom_precision": custom_precision,
                },
            )

    engine.register_metric("configurable", CustomConfigurableMetric)
    engine.add_metric("configurable")

    results = engine.run_benchmark(num_samples=20, warmup_runs=2)
    print("Configurable Custom Results:")
    engine.print_results()


def demo_task_specific_configurations():
    """Demonstrate different task-specific configurations."""

    print("\nTask-Specific Configuration Examples")

    # Binary classification with custom threshold
    binary_classification = ClassificationMetric(
        task_type="binary-classification",
        average="binary",
        include_confusion_matrix=True,
        custom_threshold=0.7,
    )

    # Multi-class classification with macro averaging
    multiclass_classification = ClassificationMetric(
        task_type="multiclass-classification",
        average="macro",
        include_confusion_matrix=True,
    )

    # Regression with all metrics enabled
    full_regression = RegressionMetric(
        task_type="regression", include_r2=True, include_mae=True, include_mse=True
    )

    print("Available configurations:")
    print(f"- Binary Classification: {binary_classification.task_type}")
    print(f"- Multi-class Classification: {multiclass_classification.task_type}")
    print(f"- Full Regression: {full_regression.task_type}")


if __name__ == "__main__":
    main()
    # demo_task_specific_configurations()
