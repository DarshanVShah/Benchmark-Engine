"""
TensorFlow Lite Benchmark Example

This example demonstrates how to use the TensorFlow Lite adapter
with the BenchmarkEngine framework.
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
    print("TENSORFLOW LITE TEXT CLASSIFICATION BENCHMARK")
    
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
    engine.load_dataset("sst2", "sst2")
    
    # 4. Load TFLite model
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
    results = engine.run_benchmark()
    
    # 7. Display results
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
    print("TENSORFLOW LITE IMAGE CLASSIFICATION BENCHMARK")
    
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



def main():
    """Main function demonstrating TFLite adapter usage."""
    
    # Run text classification benchmark
    results1 = benchmark_tflite_text_classification()
    
    # Run image classification benchmark (demo only)
    results2 = benchmark_tflite_image_classification()
    
    # Summary  
    print("\nNext Steps:")
    print("- Use actual TFLite model files")
    print("- Add image dataset adapter")
    print("- Test with real-world models")
    print("- Optimize for specific use cases")


if __name__ == "__main__":
    main() 