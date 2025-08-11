
# BenchmarkEngine

A simple framework for benchmarking machine learning models with a plugin-based architecture.

## What is this?

BenchmarkEngine lets you easily test and compare different machine learning models using a consistent interface. It's designed to be extensible - you can add support for new model types, metrics, and datasets as plugins.

## Quick Start

```python
from core import BenchmarkEngine
from plugins import DummyModelAdapter, DummyAccuracyMetric, DummyDataset

# Create the benchmark engine
engine = BenchmarkEngine()

# Register your components
engine.register_adapter("dummy", DummyModelAdapter)
engine.register_metric("accuracy", DummyAccuracyMetric)
engine.register_dataset("dummy", DummyDataset)

# Load your model and dataset
engine.load_model("dummy", "path/to/model")
engine.load_dataset("dummy", "path/to/dataset")

# Add metrics to measure
engine.add_metric("accuracy")

# Run the benchmark
results = engine.run_benchmark(num_samples=20)

# Export results
engine.export_results("results.json", format="json")
```

## Running Examples

```bash
# Basic benchmark
python examples/basic_benchmark.py

```

## Creating Custom Plugins

The framework uses a plugin architecture, so you can easily add support for:
- New model types (HuggingFace, TensorFlow Lite, ONNX, etc.)
- New metrics (accuracy, latency, memory usage, etc.)
- New datasets (ImageNet, GLUE, custom datasets, etc.)

## Why this approach?

- **Simple**: Easy to get started with dummy plugins
- **Extensible**: Add new capabilities without changing the core
- **Consistent**: Same interface regardless of which plugins you use
- **Community-friendly**: Others can easily contribute new plugins 
