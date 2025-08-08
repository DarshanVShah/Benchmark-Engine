"""
Refactored Framework Example

This example demonstrates the new architecture:
- Template Method pattern in BaseModelAdapter (preprocess -> run -> postprocess)
- Explicit input/output type contracts
- Compatibility validation between datasets and model adapters
- Framework acts as exam administrator, user provides model (student) with adapter (pencil)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import BenchmarkEngine
from core.interfaces import BaseModelAdapter, DataType, OutputType, ModelType
from benchmark_datasets.simple_text_dataset import SimpleTextDataset
from metrics.simple_accuracy_metric import SimpleAccuracyMetric


class DummyTextClassifier(BaseModelAdapter):
    """
    Dummy text classifier that demonstrates the Template Method pattern.
    
    This adapter shows how a user would wrap their model to work with the framework.
    The framework provides the template (predict method), user fills in the specifics.
    """
    
    def __init__(self):
        self.model_loaded = False
        self.model_name = "DummyTextClassifier"
    
    @property
    def input_type(self) -> DataType:
        """This model expects text input."""
        return DataType.TEXT
    
    @property
    def output_type(self) -> OutputType:
        """This model outputs class IDs."""
        return OutputType.CLASS_ID
    
    def load(self, model_path: str) -> bool:
        """
        Load the model (dummy implementation).
        
        In a real implementation, this would load the actual model from the path.
        """
        print(f"Loading dummy text classifier from {model_path}")
        self.model_loaded = True
        print("  ✓ Dummy model loaded successfully")
        return True
    
    def configure(self, config: dict) -> bool:
        """Configure the model (dummy implementation)."""
        print(f"Configuring dummy model with: {config}")
        return True
    
    def preprocess(self, raw_input: any) -> any:
        """
        Preprocess text input.
        
        This is where the user implements their specific preprocessing logic.
        """
        if not isinstance(raw_input, str):
            print(f"Warning: Expected string input, got {type(raw_input)}")
            return None
        
        # Simple preprocessing: lowercase and basic cleaning
        processed = raw_input.lower().strip()
        return processed
    
    def run(self, preprocessed_input: any) -> any:
        """
        Run inference on preprocessed input.
        
        This is where the user implements their model's inference logic.
        """
        if preprocessed_input is None:
            return None
        
        # Dummy classification logic
        # In a real implementation, this would call the actual model
        positive_words = ['great', 'love', 'amazing', 'fantastic', 'delicious', 'happy']
        negative_words = ['terrible', 'poor', 'boring', 'disappointed']
        
        text = preprocessed_input
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        # Simple rule-based classification
        if positive_count > negative_count:
            return [1]  # Positive class
        elif negative_count > positive_count:
            return [0]  # Negative class
        else:
            return [1]  # Default to positive
    
    def postprocess(self, model_output: any) -> any:
        """
        Postprocess model output to standardized format.
        
        This is where the user converts their model's raw output to the expected format.
        """
        if model_output is None:
            return None
        
        # Ensure output is in the expected format (list of class IDs)
        if isinstance(model_output, list):
            return model_output
        elif isinstance(model_output, (int, float)):
            return [int(model_output)]
        else:
            return [0]  # Default fallback
    
    def get_model_info(self) -> dict:
        """Return metadata about the model."""
        return {
            "name": self.model_name,
            "type": "dummy_text_classifier",
            "loaded": self.model_loaded,
            "input_type": self.input_type.value,
            "output_type": self.output_type.value
        }
    
    def get_model_type(self) -> ModelType:
        """Return the model type."""
        return ModelType.CUSTOM


def demonstrate_compatibility_validation():
    """Demonstrate how the framework validates compatibility."""
    print("\n" + "="*60)
    print("COMPATIBILITY VALIDATION DEMONSTRATION")
    print("="*60)
    
    # Create engine
    engine = BenchmarkEngine()
    
    # Register components
    engine.register_adapter("dummy", DummyTextClassifier)
    engine.register_dataset("simple_text", SimpleTextDataset)
    engine.register_metric("simple_accuracy", SimpleAccuracyMetric)
    
    # Test 1: Compatible setup (should work)
    print("\nTest 1: Compatible Setup")
    print("-" * 30)
    
    # Load model
    success = engine.load_model("dummy", "dummy_model_path")
    print(f"Model loading: {'✓' if success else '✗'}")
    
    # Load dataset
    success = engine.load_dataset("simple_text", "test_samples.txt")
    print(f"Dataset loading: {'✓' if success else '✗'}")
    
    # Add metric
    success = engine.add_metric("simple_accuracy")
    print(f"Metric addition: {'✓' if success else '✗'}")
    
    # Validate setup
    success = engine.validate_setup()
    print(f"Setup validation: {'✓' if success else '✗'}")
    
    if success:
        print("✓ All components are compatible!")
        print(f"  - Dataset outputs: {engine.dataset.output_type.value}")
        print(f"  - Model expects: {engine.model_adapter.input_type.value}")
        print(f"  - Model outputs: {engine.model_adapter.output_type.value}")
        print(f"  - Metric expects: {engine.metrics[0].expected_input_type.value}")
    
    return engine


def run_benchmark_demonstration(engine: BenchmarkEngine):
    """Run a complete benchmark to demonstrate the framework."""
    print("\n" + "="*60)
    print("BENCHMARK DEMONSTRATION")
    print("="*60)
    
    try:
        # Configure benchmark
        engine.configure_benchmark({
            "num_samples": 5,  # Test on 5 samples
            "warmup_runs": 2,
            "batch_size": 1,
            "precision": "fp32",
            "device": "cpu"
        })
        
        # Run benchmark
        print("\nRunning benchmark...")
        results = engine.run_benchmark()
        
        # Print results
        engine.print_results()
        
        return results
        
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        return None


def demonstrate_template_method_pattern():
    """Demonstrate the Template Method pattern in action."""
    print("\n" + "="*60)
    print("TEMPLATE METHOD PATTERN DEMONSTRATION")
    print("="*60)
    
    # Create model adapter
    model = DummyTextClassifier()
    model.load("dummy_path")
    
    # Test the complete pipeline
    test_texts = [
        "This is a great movie!",
        "The service was very poor.",
        "I love this product.",
        "This book is really boring."
    ]
    
    print("\nTesting Template Method pattern:")
    print("raw_input → preprocess → run → postprocess → prediction")
    print("-" * 60)
    
    for text in test_texts:
        print(f"\nInput: '{text}'")
        
        # Demonstrate each step
        preprocessed = model.preprocess(text)
        print(f"  Preprocessed: '{preprocessed}'")
        
        raw_output = model.run(preprocessed)
        print(f"  Raw output: {raw_output}")
        
        prediction = model.postprocess(raw_output)
        print(f"  Final prediction: {prediction}")
        
        # Test the complete predict() method
        complete_prediction = model.predict(text)
        print(f"  Complete predict(): {complete_prediction}")


def main():
    """Main demonstration function."""
    print("🚀 REFACTORED BENCHMARKING FRAMEWORK DEMONSTRATION")
    print("="*60)
    print("\nThis example demonstrates:")
    print("1. Template Method pattern in BaseModelAdapter")
    print("2. Explicit input/output type contracts")
    print("3. Compatibility validation between components")
    print("4. Framework as exam administrator, user provides model with adapter")
    
    # Demonstrate Template Method pattern
    demonstrate_template_method_pattern()
    
    # Demonstrate compatibility validation
    engine = demonstrate_compatibility_validation()
    
    # Run benchmark demonstration
    if engine:
        results = run_benchmark_demonstration(engine)
        
        if results:
            print("\n🎉 SUCCESS! Framework is working correctly.")
            print("\nKey features demonstrated:")
            print("✓ Template Method pattern (preprocess → run → postprocess)")
            print("✓ Explicit type contracts (TEXT → CLASS_ID)")
            print("✓ Compatibility validation")
            print("✓ Modular architecture")
        else:
            print("\n❌ Benchmark failed. Check the error messages above.")
    else:
        print("\n❌ Setup failed. Check the error messages above.")


if __name__ == "__main__":
    main()
