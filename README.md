# BenchmarkEngine

A flexible and modular benchmarking framework for evaluating machine learning models across different datasets and metrics.

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd BenchmarkEngine
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage

The framework follows a simple pattern:
- **BenchmarkEngine**: Orchestrates the benchmarking process
- **Model Adapters**: Wrap your models to work with the framework
- **Datasets**: Provide standardized test data
- **Metrics**: Evaluate model performance

#### Example: Test a TFLite Model

```python
from core.engine import BenchmarkEngine
from plugins.tflite_adapter import TensorFlowLiteAdapter
from benchmark_datasets.template_dataset import TemplateDataset
from metrics.template_metric import TemplateAccuracyMetric

# Setup
engine = BenchmarkEngine(
    model_adapter=TensorFlowLiteAdapter("models/your_model.tflite"),
    dataset=TemplateDataset("path/to/dataset.txt"),
    metrics=[TemplateAccuracyMetric()]
)

# Run benchmark
results = engine.run_benchmark(num_samples=100)
print(f"Accuracy: {results['accuracy']:.4f}")
```

## 📁 Project Structure

```
BenchmarkEngine/
├── core/                    # Core framework components
│   ├── engine.py           # Main benchmarking engine
│   ├── interfaces.py       # Abstract base classes
│   ├── dataset_registry.py # Dataset management
│   └── types.py           # Type definitions
├── plugins/                # Model adapters
│   ├── tflite_adapter.py  # TensorFlow Lite support
│   ├── huggingface_adapter.py # HuggingFace models
│   └── dummy_adapter.py   # Testing adapter
├── benchmark_datasets/     # Dataset implementations
│   └── template_dataset.py # Flexible dataset wrapper
├── metrics/                # Evaluation metrics
│   ├── template_metric.py # Flexible accuracy metric
│   └── task_metric.py     # Task-specific metrics
├── examples/               # Usage examples
└── models/                 # Your model files
```

## 🔧 Key Features

- **Modular Design**: Easy to add new models, datasets, and metrics
- **Type Safety**: Explicit input/output type contracts
- **Compatibility Validation**: Automatic validation of model-dataset-metric compatibility
- **Flexible Data Handling**: Support for various data formats and task types
- **Template Method Pattern**: Clean separation of preprocessing, inference, and postprocessing

## 📊 Supported Task Types

- **Emotion Classification**: Multi-label and single-label emotion detection
- **Sentiment Analysis**: Binary and multi-class sentiment classification
- **Text Classification**: General text categorization tasks
- **Image Classification**: Computer vision tasks (via TFLite)

## 🎯 Examples

### Quick TFLite Test
```bash
python examples/tflite_quick_test.py
```

### Comprehensive Emotion Testing
```bash
python examples/tflite_emotion_comprehensive_test.py
```

## 🤝 Adding Your Own Components

### Custom Model Adapter
```python
from core.interfaces import BaseModelAdapter, DataType, OutputType

class MyModelAdapter(BaseModelAdapter):
    @property
    def input_type(self):
        return DataType.TEXT
    
    @property
    def output_type(self):
        return OutputType.CLASS_ID
    
    def preprocess(self, raw_input):
        # Your preprocessing logic
        return processed_input
    
    def run(self, preprocessed_input):
        # Your model inference
        return model_output
    
    def postprocess(self, model_output):
        # Your postprocessing logic
        return final_output
```

### Custom Dataset
```python
from core.interfaces import BaseDataset, DataType

class MyDataset(BaseDataset):
    @property
    def output_type(self):
        return DataType.TEXT
    
    def load_data(self):
        # Load your data
        pass
    
    def get_samples(self, num_samples=None):
        # Return data samples
        pass
```

### Custom Metric
```python
from core.interfaces import BaseMetric, OutputType

class MyMetric(BaseMetric):
    @property
    def expected_input_type(self):
        return OutputType.CLASS_ID
    
    def calculate(self, predictions, targets):
        # Your metric calculation
        return metric_value
```

## 📝 Requirements

- Python 3.8+
- TensorFlow 2.x (for TFLite support)
- PyTorch (for HuggingFace support)
- pandas
- numpy

## 🤔 Need Help?

Check the examples directory for working implementations, or examine the core interfaces to understand the framework architecture.

## 📄 License

[Your License Here] 