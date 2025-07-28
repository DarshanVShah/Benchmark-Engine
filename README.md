# BenchmarkEngine Framework

A flexible, extensible framework for model benchmarking with plugin-based architecture.

## 🎯 Overview

BenchmarkEngine is designed as a **framework**, not just a tool. Like highlight.js defines language contracts that allow community contributions, this framework defines abstract interfaces that enable seamless integration of new model types, metrics, and datasets.

### Core Philosophy

- **Stable Core API**: The orchestration layer remains stable while plugins evolve
- **Plugin Architecture**: Model adapters, metrics, and datasets are pluggable components
- **Community-Driven**: Easy to contribute new adapters and metrics
- **Framework Mindset**: Designed for extensibility, not just immediate use cases

## 🏗️ Architecture

```
BenchmarkEngine/
├── core/                    # Core framework (stable API)
│   └── __init__.py         # BenchmarkEngine + abstract interfaces
├── plugins/                 # Plugin implementations
│   ├── __init__.py         # Plugin registry
│   ├── dummy_adapter.py    # Example model adapter
│   ├── dummy_metric.py     # Example metric
│   └── dummy_dataset.py    # Example dataset
├── examples/               # Usage examples
│   ├── basic_benchmark.py  # Basic usage
│   └── custom_plugins.py   # Custom plugin creation
└── README.md              # This file
```

### Core Components

1. **BenchmarkEngine**: The orchestration layer that handles `load → run → collect → report`
2. **BaseModelAdapter**: Abstract interface for model adapters (HuggingFace, TensorFlow Lite, ONNX, etc.)
3. **BaseMetric**: Abstract interface for metrics (accuracy, latency, quantization degradation, etc.)
4. **BaseDataset**: Abstract interface for dataset loaders

## 🚀 Quick Start

### Basic Usage

```python
from core import BenchmarkEngine
from plugins import DummyModelAdapter, DummyAccuracyMetric, DummyDataset

# Create engine
engine = BenchmarkEngine()

# Register plugins
engine.register_adapter("dummy", DummyModelAdapter)
engine.register_metric("accuracy", DummyAccuracyMetric)
engine.register_dataset("dummy", DummyDataset)

# Load model and dataset
engine.load_model("dummy", "path/to/model")
engine.load_dataset("dummy", "path/to/dataset")

# Add metrics
engine.add_metric("accuracy")

# Run benchmark
results = engine.run_benchmark(num_samples=20)

# Export results
engine.export_results("results.json", format="json")
engine.export_results("results.md", format="markdown")
```

### Running Examples

```bash
# Basic benchmark with dummy plugins
python examples/basic_benchmark.py

# Custom plugins demonstration
python examples/custom_plugins.py
```

## 🔌 Creating Custom Plugins

### Model Adapter

```python
from core import BaseModelAdapter
from typing import Dict, Any

class MyHuggingFaceAdapter(BaseModelAdapter):
    def __init__(self):
        self.model = None
        
    def load(self, model_path: str) -> bool:
        # Load your model here
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_path)
        return True
    
    def run(self, inputs: Any) -> Any:
        # Run inference
        return self.model(inputs)
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "MyModel",
            "type": "huggingface",
            "parameters": 1000000
        }
```

### Custom Metric

```python
from core import BaseMetric
from typing import Dict, Any

class MyLatencyMetric(BaseMetric):
    def calculate(self, predictions: Any, targets: Any, **kwargs) -> Dict[str, float]:
        # Calculate your metrics here
        return {
            "avg_latency_ms": 25.5,
            "p95_latency_ms": 45.2
        }
    
    def get_name(self) -> str:
        return "MyLatency"
```

### Custom Dataset

```python
from core import BaseDataset
from typing import Dict, Any, List, Optional

class MyImageDataset(BaseDataset):
    def load(self, dataset_path: str) -> bool:
        # Load your dataset here
        return True
    
    def get_samples(self, num_samples: Optional[int] = None) -> List[Any]:
        # Return your samples
        return [{"image": "data", "label": 1}]
    
    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "name": "MyDataset",
            "type": "image",
            "num_samples": 1000
        }
```

## 🎨 Framework Design Principles

### 1. Stable Core API
The `BenchmarkEngine` class provides a consistent interface regardless of which plugins are used:

```python
# Same API, different plugins
engine.load_model("huggingface", "bert-base-uncased")
engine.load_model("tflite", "model.tflite")
engine.load_model("onnx", "model.onnx")
```

### 2. Plugin Registration System
Dynamic registration allows runtime plugin discovery:

```python
# Register plugins
engine.register_adapter("huggingface", HuggingFaceAdapter)
engine.register_adapter("tflite", TFLiteAdapter)

# Use by name
engine.load_model("huggingface", model_path)
```

### 3. Extensible Metrics
Add new metrics without modifying the core:

```python
engine.register_metric("accuracy", AccuracyMetric)
engine.register_metric("latency", LatencyMetric)
engine.register_metric("memory", MemoryMetric)

# Use multiple metrics
engine.add_metric("accuracy")
engine.add_metric("latency")
```

### 4. Community-Ready Architecture
Like highlight.js, the framework is designed for community contributions:

- **Clear Contracts**: Abstract interfaces define what plugins must implement
- **Versioned Support**: Official plugins can be versioned and maintained
- **Easy Integration**: New plugins integrate seamlessly without core changes

## 📊 Output Formats

The framework supports multiple output formats:

### JSON Output
```json
{
  "model_info": {
    "name": "DummyModel-1234",
    "type": "dummy",
    "size_mb": 45
  },
  "timing": {
    "total_time": 2.5,
    "average_inference_time": 0.0125,
    "throughput": 80.0
  },
  "metrics": {
    "DummyAccuracy": {
      "accuracy": 0.8542,
      "precision": 0.8234,
      "recall": 0.8765
    }
  }
}
```

### Markdown Output
```markdown
# Benchmark Results

## Model Information
- **name**: DummyModel-1234
- **type**: dummy
- **size_mb**: 45

## Timing Results
- **Total Time**: 2.50s
- **Average Inference Time**: 0.0125s
- **Throughput**: 80.00 samples/s

## Metrics
### DummyAccuracy
- **accuracy**: 0.8542
- **precision**: 0.8234
- **recall**: 0.8765
```

## 🔮 Future Extensions

The framework is designed to evolve gracefully:

### Planned Adapters
- **HuggingFace**: Full transformer model support
- **TensorFlow Lite**: Mobile/edge deployment
- **ONNX Runtime**: Cross-platform inference
- **PyTorch**: Native PyTorch models
- **TensorRT**: NVIDIA GPU optimization

### Planned Metrics
- **Accuracy**: Classification/regression accuracy
- **Latency**: Inference timing statistics
- **Memory**: Memory usage profiling
- **Throughput**: Samples per second
- **Quantization**: Model compression analysis

### Planned Datasets
- **ImageNet**: Computer vision benchmarks
- **GLUE**: NLP benchmarks
- **Custom**: User-defined datasets

## 🤝 Contributing

The framework follows the highlight.js model:

1. **Implement Interfaces**: Create plugins that implement the abstract interfaces
2. **Test Integration**: Ensure your plugin works with the core framework
3. **Document Usage**: Provide clear examples and documentation
4. **Community Review**: Submit for inclusion in the official plugin registry

### Plugin Guidelines

- **Follow Contracts**: Implement all required methods from abstract interfaces
- **Handle Errors**: Graceful error handling and meaningful error messages
- **Document Dependencies**: Clearly specify any external dependencies
- **Provide Examples**: Include usage examples in your plugin documentation

## 📚 Documentation

- **Core API**: `core/__init__.py` - Main framework documentation
- **Plugin Examples**: `plugins/` - Reference implementations
- **Usage Examples**: `examples/` - Working demonstrations
- **Architecture Guide**: This README - Framework design principles

## 🎯 Why This Approach?

This framework design enables:

1. **Rapid Prototyping**: Dummy plugins let you test the framework immediately
2. **Incremental Development**: Add real plugins one at a time
3. **Community Growth**: Easy for others to contribute new capabilities
4. **Long-term Evolution**: Core remains stable while ecosystem expands
5. **Production Ready**: Framework scales from prototype to production

The key insight is that **an engineer is the integral of code over time** - good frameworks are designed so that adding new features doesn't require rewriting the core, and the code evolves gracefully as the ecosystem grows. 