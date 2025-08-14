
# BenchmarkEngine

A standardized benchmarking framework for machine learning models. 

## Architecture

- **BenchmarkEngine**: The "exam administrator" that provides standardized testing infrastructure
- **Standard Datasets**: Curated benchmark datasets for different ML tasks
- **User Models**: Your models wrapped in adapters that conform to our interface
- **Adapters**: Must implement `BaseModelAdapter` interface for compatibility

## Supported Task Types

- **Emotion Classification**: Multi-label and single-label emotion detection
- **Sentiment Analysis**: Text sentiment classification
- **Text Classification**: General text categorization
- **Image Classification**: Computer vision tasks
- **Object Detection**: Object localization and classification

Currently only emotion classifiers.

## Emotion Standardization

The BenchmarkEngine includes a comprehensive emotion standardization system that maps emotions from different models and datasets to a consistent set of 32 standardized emotions. This ensures:

- **Human-readable output**: Get emotion names instead of just numbers
- **Consistent comparison**: Compare models using the same emotion labels
- **Multi-scheme support**: Works with 2018-E-c-En, GoEmotions, and common model outputs
- **Confidence scoring**: Understand how reliable the emotion mapping is

See [EMOTION_STANDARDIZATION.md](docs/EMOTION_STANDARDIZATION.md) for detailed documentation and examples.

## Using Your Own Models

### 1. Create an Adapter
Your adapter must inherit from `BaseModelAdapter` and implement:

```python
from core.interfaces import BaseModelAdapter, DataType, OutputType

class MyModelAdapter(BaseModelAdapter):
    @property
    def input_type(self) -> DataType:
        return DataType.TEXT  # or IMAGE, etc.
    
    @property
    def output_type(self) -> OutputType:
        return OutputType.CLASS_ID  # or PROBABILITIES, etc.
    
    def preprocess(self, raw_input: Any) -> Any:
        # Convert input to model format
        pass
    
    def run(self, preprocessed_input: Any) -> Any:
        # Run model inference
        pass
    
    def postprocess(self, model_output: Any) -> Any:
        # Convert output to standard format
        pass
```

### 2. Use the Framework
```python
from core.engine import BenchmarkEngine

# Create engine
engine = BenchmarkEngine()

# Register your adapter
engine.register_adapter("mymodel", MyModelAdapter)

# Load your model
engine.load_model("mymodel", "path/to/model", config)

# Run benchmarks on our standard datasets
results = engine.run_benchmark()
```

## Key Features

- **Standardized Testing**: Consistent evaluation across different models
- **Real Datasets**: Curated benchmark datasets for reliable testing
- **Model Agnostic**: Works with any model through adapters
- **Type Safety**: Explicit input/output type contracts
- **Performance Metrics**: Comprehensive evaluation including accuracy and timing

