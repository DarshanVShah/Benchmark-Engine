"""
HuggingFace Adapter Example with Refactored Framework

This example demonstrates how to use the HuggingFace adapter with the new architecture:
- Template Method pattern (preprocess -> run -> postprocess)
- Explicit type contracts
- Compatibility validation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import BenchmarkEngine
from core.interfaces import DataType, OutputType
from plugins.huggingface_adapter import HuggingFaceAdapter
from benchmark_datasets.simple_text_dataset import SimpleTextDataset
from metrics.simple_accuracy_metric import SimpleAccuracyMetric


def demonstrate_huggingface_adapter():
    """Demonstrate the HuggingFace adapter with the refactored framework."""
    print("🤗 HUGGINGFACE ADAPTER WITH REFACTORED FRAMEWORK")
    print("="*60)
    
    # Create engine
    engine = BenchmarkEngine()
    
    # Register components
    engine.register_adapter("huggingface", HuggingFaceAdapter)
    engine.register_dataset("simple_text", SimpleTextDataset)
    engine.register_metric("simple_accuracy", SimpleAccuracyMetric)
    
    print("\n1. Loading HuggingFace Model")
    print("-" * 30)
    
    # Load a small HuggingFace model for testing
    # Using a small model to avoid long download times
    model_path = "distilbert-base-uncased"
    
    success = engine.load_model("huggingface", model_path, {
        "device": "cpu",
        "precision": "fp32",
        "max_length": 128
    })
    
    if not success:
        print("❌ Failed to load HuggingFace model")
        print("Note: This requires transformers and torch to be installed")
        print("Install with: pip install transformers torch")
        return None
    
    print("\n2. Loading Dataset")
    print("-" * 30)
    
    success = engine.load_dataset("simple_text", "test_samples.txt")
    if not success:
        print("❌ Failed to load dataset")
        return None
    
    print("\n3. Adding Metric")
    print("-" * 30)
    
    success = engine.add_metric("simple_accuracy")
    if not success:
        print("❌ Failed to add metric")
        return None
    
    print("\n4. Validating Compatibility")
    print("-" * 30)
    
    success = engine.validate_setup()
    if success:
        print("✓ All components are compatible!")
        print(f"  - Dataset outputs: {engine.dataset.output_type.value}")
        print(f"  - Model expects: {engine.model_adapter.input_type.value}")
        print(f"  - Model outputs: {engine.model_adapter.output_type.value}")
        print(f"  - Metric expects: {engine.metrics[0].expected_input_type.value}")
    else:
        print("❌ Compatibility validation failed")
        return None
    
    return engine


def run_huggingface_benchmark(engine: BenchmarkEngine):
    """Run benchmark with HuggingFace model."""
    print("\n5. Running Benchmark")
    print("-" * 30)
    
    try:
        # Configure benchmark
        engine.configure_benchmark({
            "num_samples": 3,  # Test on 3 samples to avoid long runtime
            "warmup_runs": 1,
            "batch_size": 1,
            "precision": "fp32",
            "device": "cpu"
        })
        
        # Run benchmark
        print("Running benchmark...")
        results = engine.run_benchmark()
        
        # Print results
        engine.print_results()
        
        return results
        
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        return None


def demonstrate_template_method_with_huggingface():
    """Demonstrate Template Method pattern with HuggingFace adapter."""
    print("\n" + "="*60)
    print("TEMPLATE METHOD WITH HUGGINGFACE")
    print("="*60)
    
    # Create HuggingFace adapter
    adapter = HuggingFaceAdapter()
    
    # Load model
    model_path = "distilbert-base-uncased"
    success = adapter.load(model_path)
    
    if not success:
        print("❌ Failed to load HuggingFace model for demonstration")
        print("Note: This requires transformers and torch to be installed")
        return
    
    # Configure adapter
    adapter.configure({
        "device": "cpu",
        "precision": "fp32",
        "max_length": 64
    })
    
    # Test the Template Method pattern
    test_texts = [
        "This is a great movie!",
        "The service was very poor.",
        "I love this product."
    ]
    
    print("\nTesting Template Method pattern with HuggingFace:")
    print("raw_input → preprocess → run → postprocess → prediction")
    print("-" * 60)
    
    for text in test_texts:
        print(f"\nInput: '{text}'")
        
        # Demonstrate each step
        preprocessed = adapter.preprocess(text)
        print(f"  Preprocessed: Tokenized input (shape: {preprocessed['input_ids'].shape})")
        
        raw_output = adapter.run(preprocessed)
        print(f"  Raw output: Logits (shape: {raw_output.shape})")
        
        prediction = adapter.postprocess(raw_output)
        print(f"  Final prediction: {prediction}")
        
        # Test the complete predict() method
        complete_prediction = adapter.predict(text)
        print(f"  Complete predict(): {complete_prediction}")


def main():
    """Main demonstration function."""
    print("🚀 HUGGINGFACE ADAPTER WITH REFACTORED FRAMEWORK")
    print("="*60)
    print("\nThis example demonstrates:")
    print("1. HuggingFace adapter with Template Method pattern")
    print("2. Explicit type contracts (TEXT → CLASS_ID)")
    print("3. Compatibility validation")
    print("4. Real model inference with transformers")
    
    # Demonstrate Template Method with HuggingFace
    demonstrate_template_method_with_huggingface()
    
    # Demonstrate full framework with HuggingFace
    engine = demonstrate_huggingface_adapter()
    
    # Run benchmark demonstration
    if engine:
        results = run_huggingface_benchmark(engine)
        
        if results:
            print("\n🎉 SUCCESS! HuggingFace adapter is working correctly.")
            print("\nKey features demonstrated:")
            print("✓ Template Method pattern with real model")
            print("✓ Explicit type contracts")
            print("✓ Compatibility validation")
            print("✓ Real inference with transformers")
        else:
            print("\n❌ Benchmark failed. Check the error messages above.")
    else:
        print("\n❌ Setup failed. Check the error messages above.")


if __name__ == "__main__":
    main()
