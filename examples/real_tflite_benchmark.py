"""
Real TensorFlow Lite Benchmark

This example demonstrates a real benchmark using:
- Actual TFLite model: models/notQuantizedModel.tflite
- Real dataset: benchmark_datasets/localTestSets/2018-E-c-En-test-gold.txt
- Emotion detection task with 11 emotion categories
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import BenchmarkEngine, DataType, OutputType
from plugins import TensorFlowLiteAdapter
from metrics import AccuracyMetric
from benchmark_datasets import LocalEmotionDataset


def benchmark_real_tflite_emotion():
    """
    Real benchmark using actual TFLite model and emotion dataset.
    """
    print("=" * 60)
    print("REAL TENSORFLOW LITE EMOTION DETECTION BENCHMARK")
    print("=" * 60)
    print("Model: notQuantizedModel.tflite (418MB)")
    print("Dataset: 2018-E-c-En-test-gold.txt (3,261 samples)")
    print("Task: Multi-label emotion detection (11 emotions)")
    print("Emotions: anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust")
    print()
    
    # User declares everything in main function
    engine = BenchmarkEngine()
    
    # 1. Register components
    engine.register_adapter("tflite", TensorFlowLiteAdapter)
    engine.register_metric("accuracy", AccuracyMetric)
    engine.register_dataset("emotion", LocalEmotionDataset)
    
    # 2. Configure benchmark
    engine.configure_benchmark({
        "num_samples": 100,  # Reasonable sample size for real model
        "warmup_runs": 3,
        "batch_size": 1,
        "precision": "fp32",
        "device": "cpu"
    })
    
    # 3. Load dataset
    print("Loading emotion detection dataset...")
    dataset_path = "benchmark_datasets/localTestSets/2018-E-c-En-test-gold.txt"
    engine.load_dataset("emotion", dataset_path)
    
    # 4. Load TFLite model
    print("Loading TensorFlow Lite model...")
    model_path = "models/notQuantizedModel.tflite"
    engine.load_model(
        "tflite", 
        model_path,
        {
            "task_type": "emotion-detection",
            "max_length": 512,
            # Note: This model might need specific tokenizer configuration
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
    
    # 8. Analyze emotion-specific performance
    print("\n" + "=" * 60)
    print("EMOTION-SPECIFIC ANALYSIS")
    print("=" * 60)
    
    # Get dataset info for emotion statistics
    dataset_info = engine.dataset.get_dataset_info()
    emotion_stats = dataset_info.get("emotion_statistics", {})
    
    print("Dataset Emotion Distribution:")
    for emotion, count in emotion_stats.items():
        percentage = (count / len(engine.dataset.data)) * 100
        print(f"  {emotion:12}: {count:4d} samples ({percentage:5.1f}%)")
    
    print("\nModel Performance:")
    if "metrics" in results and "Accuracy" in results["metrics"]:
        accuracy = results["metrics"]["Accuracy"]["accuracy"]
        print(f"  Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Show detailed metrics if available
        if "precision" in results["metrics"]["Accuracy"]:
            precision = results["metrics"]["Accuracy"]["precision"]
            recall = results["metrics"]["Accuracy"]["recall"]
            f1 = results["metrics"]["Accuracy"]["f1_score"]
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1-Score: {f1:.3f}")
    
    return results


def analyze_model_compatibility():
    """
    Analyze the TFLite model's input/output requirements.
    """
    print("\n" + "=" * 60)
    print("MODEL COMPATIBILITY ANALYSIS")
    print("=" * 60)
    
    # Load model to analyze its structure
    adapter = TensorFlowLiteAdapter()
    model_path = "models/notQuantizedModel.tflite"
    
    if adapter.load(model_path):
        print("✅ Model loaded successfully")
        
        # Get model info
        model_info = adapter.get_model_info()
        print(f"Model Size: {model_info.get('model_size_mb', 0):.1f} MB")
        print(f"Input Shape: {model_info.get('input_shape', 'Unknown')}")
        print(f"Output Shape: {model_info.get('output_shape', 'Unknown')}")
        print(f"Task Type: {model_info.get('task_type', 'Unknown')}")
        
        # Analyze input/output compatibility
        print(f"\nData Contract Analysis:")
        print(f"  Model Input Type: {adapter.input_type.value}")
        print(f"  Model Output Type: {adapter.output_type.value}")
        print(f"  Dataset Output Type: {DataType.TEXT.value}")
        
        if adapter.input_type == DataType.TEXT:
            print("  ✅ Input compatibility: Model expects text, dataset provides text")
        else:
            print("  ⚠️  Input compatibility: Type mismatch")
            
    else:
        print("❌ Failed to load model for analysis")


def demonstrate_workflow():
    """
    Demonstrate the complete user workflow for real benchmarks.
    """
    print("\n" + "=" * 60)
    print("COMPLETE USER WORKFLOW")
    print("=" * 60)
    
    print("1. Create engine instance")
    print("   engine = BenchmarkEngine()")
    print()
    
    print("2. Register components")
    print("   engine.register_adapter('tflite', TensorFlowLiteAdapter)")
    print("   engine.register_metric('accuracy', AccuracyMetric)")
    print("   engine.register_dataset('emotion', LocalEmotionDataset)")
    print()
    
    print("3. Configure benchmark parameters")
    print("   engine.configure_benchmark({")
    print("       'num_samples': 100,")
    print("       'warmup_runs': 3,")
    print("       'batch_size': 1,")
    print("       'precision': 'fp32',")
    print("       'device': 'cpu'")
    print("   })")
    print()
    
    print("4. Load dataset")
    print("   engine.load_dataset('emotion', 'path/to/dataset.txt')")
    print()
    
    print("5. Load model")
    print("   engine.load_model('tflite', 'path/to/model.tflite', config)")
    print()
    
    print("6. Add metrics")
    print("   engine.add_metric('accuracy')")
    print()
    
    print("7. Run benchmark")
    print("   results = engine.run_benchmark()")
    print()
    
    print("8. Display results")
    print("   engine.print_results()")
    print()
    
    print("✅ That's it! Framework handles everything else.")


def main():
    """Main function for real TFLite benchmark."""
    
    print("REAL TENSORFLOW LITE BENCHMARK")
    print("=" * 60)
    print("This example demonstrates a real benchmark using:")
    print("- Actual TFLite model (418MB)")
    print("- Real emotion detection dataset (3,261 samples)")
    print("- Multi-label emotion classification")
    print()
    
    # Check if files exist
    model_path = "models/notQuantizedModel.tflite"
    dataset_path = "benchmark_datasets/localTestSets/2018-E-c-En-test-gold.txt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure the TFLite model is in the models/ directory")
        return
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        print("Please ensure the emotion dataset is in the benchmark_datasets/localTestSets/ directory")
        return
    
    print("✅ All required files found")
    print()
    
    # Analyze model compatibility
    analyze_model_compatibility()
    
    # Run the real benchmark
    results = benchmark_real_tflite_emotion()
    
    # Show workflow
    demonstrate_workflow()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✅ Real TFLite model benchmark completed")
    print("✅ Actual emotion detection dataset used")
    print("✅ Framework handles complex multi-label classification")
    print("✅ Performance metrics calculated")
    print("✅ Ready for production use")
    
    print("\nKey Benefits:")
    print("- Real-world model evaluation")
    print("- Actual dataset performance")
    print("- Comprehensive metrics")
    print("- Production-ready workflow")


if __name__ == "__main__":
    main() 