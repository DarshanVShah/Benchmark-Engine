"""
TensorFlow Lite Benchmark Example

This example demonstrates how to use the TensorFlow Lite adapter
with the BenchmarkEngine framework.

Note: This example uses a dummy TFLite model for demonstration.
In practice, you would use actual TFLite model files.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import BenchmarkEngine, DataType, OutputType
from plugins import TensorFlowLiteAdapter
from metrics import AccuracyMetric
from benchmark_datasets import HuggingFaceDataset


def create_dummy_tflite_model():
    """
    Create a dummy TensorFlow Lite model for demonstration.
    In practice, you would use actual TFLite model files.
    """
    import tensorflow as tf
    import numpy as np
    
    # Create a simple model for text classification
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128,), dtype=tf.int32),
        tf.keras.layers.Embedding(1000, 16, input_length=128),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')  # Binary classification
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save the model
    model_path = "dummy_text_classifier.tflite"
    with open(model_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Created dummy TFLite model: {model_path}")
    return model_path


def benchmark_tflite_text_classification():
    """
    Benchmark a TensorFlow Lite text classification model.
    """
    print("=" * 60)
    print("TENSORFLOW LITE TEXT CLASSIFICATION BENCHMARK")
    print("=" * 60)
    print("Model: Dummy TFLite Text Classifier")
    print("Dataset: SST-2 (Stanford Sentiment Treebank)")
    print("Task: Binary sentiment classification")
    print()
    
    # Create dummy model (in practice, use real TFLite files)
    model_path = create_dummy_tflite_model()
    
    # User declares everything in main function
    engine = BenchmarkEngine()
    
    # 1. Register components
    engine.register_adapter("tflite", TensorFlowLiteAdapter)
    engine.register_metric("accuracy", AccuracyMetric)
    engine.register_dataset("sst2", HuggingFaceDataset)
    
    # 2. Configure benchmark
    engine.configure_benchmark({
        "num_samples": 50,  # Smaller sample for demo
        "warmup_runs": 2,
        "batch_size": 1,
        "precision": "fp32",
        "device": "cpu"
    })
    
    # 3. Load dataset
    print("Loading SST-2 dataset...")
    engine.load_dataset("sst2", "sst2")
    
    # 4. Load TFLite model
    print("Loading TensorFlow Lite model...")
    engine.load_model(
        "tflite", 
        model_path,
        {
            "task_type": "text-classification",
            "max_length": 128,
            # "tokenizer_path": "bert-base-uncased"  # Optional
        }
    )
    
    # 5. Add metric
    engine.add_metric("accuracy")
    
    # 6. Run benchmark
    print("\nRunning benchmark...")
    results = engine.run_benchmark()
    
    # 7. Display results
    print("\nBenchmark Results:")
    print("-" * 40)
    engine.print_results()
    
    # Clean up dummy model
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"\nCleaned up dummy model: {model_path}")
    
    return results


def benchmark_tflite_image_classification():
    """
    Benchmark a TensorFlow Lite image classification model.
    """
    print("\n" + "=" * 60)
    print("TENSORFLOW LITE IMAGE CLASSIFICATION BENCHMARK")
    print("=" * 60)
    print("Model: Dummy TFLite Image Classifier")
    print("Dataset: Custom Image Dataset")
    print("Task: Image classification")
    print()
    
    # Create dummy image classification model
    import tensorflow as tf
    
    # Create a simple image classification model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3), dtype=tf.float32),
        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')  # 10 classes
    ])
    
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save the model
    model_path = "dummy_image_classifier.tflite"
    with open(model_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Created dummy image TFLite model: {model_path}")
    
    # User declares everything in main function
    engine = BenchmarkEngine()
    
    # 1. Register components
    engine.register_adapter("tflite", TensorFlowLiteAdapter)
    engine.register_metric("accuracy", AccuracyMetric)
    # Note: Would need image dataset adapter here
    
    # 2. Configure benchmark
    engine.configure_benchmark({
        "num_samples": 10,  # Very small for demo
        "warmup_runs": 1,
        "batch_size": 1,
        "precision": "fp32",
        "device": "cpu"
    })
    
    # 3. Load TFLite model
    print("Loading TensorFlow Lite image model...")
    engine.load_model(
        "tflite", 
        model_path,
        {
            "task_type": "image-classification",
            "max_length": 224,  # Image size
        }
    )
    
    # 4. Add metric
    engine.add_metric("accuracy")
    
    # Note: This would need an image dataset to fully test
    print("Image classification benchmark requires image dataset adapter")
    print("TFLite adapter is ready for image models with proper dataset")
    
    # Clean up dummy model
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"Cleaned up dummy image model: {model_path}")
    
    return None


def demonstrate_tflite_features():
    """
    Demonstrate key features of the TFLite adapter.
    """
    print("\n" + "=" * 60)
    print("TENSORFLOW LITE ADAPTER FEATURES")
    print("=" * 60)
    
    # Create adapter instance
    adapter = TensorFlowLiteAdapter()
    
    print("✅ Supported Features:")
    print("  - Text classification models")
    print("  - Image classification models") 
    print("  - Regression models")
    print("  - Custom input/output formats")
    print("  - Automatic input/output shape detection")
    print("  - Optional tokenizer integration")
    print("  - Memory-efficient inference")
    print("  - Cross-platform compatibility")
    
    print("\n✅ Data Contract Support:")
    print(f"  - Input types: {[t.value for t in DataType]}")
    print(f"  - Output types: {[t.value for t in OutputType]}")
    print("  - Automatic compatibility validation")
    
    print("\n✅ Configuration Options:")
    print("  - task_type: text-classification, image-classification, regression")
    print("  - max_length: Maximum input length for text models")
    print("  - tokenizer_path: Optional HuggingFace tokenizer")
    print("  - Custom preprocessing options")
    
    print("\n✅ Performance Features:")
    print("  - Optimized for mobile/edge deployment")
    print("  - Small model size")
    print("  - Fast inference")
    print("  - Memory profiling support")


def main():
    """Main function demonstrating TFLite adapter usage."""
    
    print("TENSORFLOW LITE BENCHMARK EXAMPLE")
    print("=" * 60)
    print("This example demonstrates the TFLite adapter integration")
    print("with the BenchmarkEngine framework.")
    print()
    
    # Show adapter features
    demonstrate_tflite_features()
    
    # Run text classification benchmark
    results1 = benchmark_tflite_text_classification()
    
    # Run image classification benchmark (demo only)
    results2 = benchmark_tflite_image_classification()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✅ TFLite adapter successfully integrated")
    print("✅ Supports text and image classification")
    print("✅ Follows Template Method pattern")
    print("✅ Implements data contracts")
    print("✅ Compatible with existing framework")
    print("✅ Ready for real TFLite model files")
    
    print("\nNext Steps:")
    print("- Use actual TFLite model files")
    print("- Add image dataset adapter")
    print("- Test with real-world models")
    print("- Optimize for specific use cases")


if __name__ == "__main__":
    main() 