"""
Multi-Dataset Benchmark Example

This example demonstrates how the framework provides standardized test suites
for different ML tasks. Just like different subjects have different exams,
different ML tasks have different standardized datasets.

For emotion classification, we test the model on multiple emotion datasets
to ensure comprehensive evaluation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import BenchmarkEngine, TaskType
from plugins import TensorFlowLiteAdapter, HuggingFaceAdapter
from metrics import TemplateAccuracyMetric, TemplateMultiLabelMetric
from benchmark_datasets import TemplateDataset


def benchmark_emotion_classification():
    """
    Benchmark emotion classification models across multiple datasets.
    
    This demonstrates the framework's ability to provide standardized test suites:
    - Different "exams" (datasets) for the same subject (emotion classification)
    - Consistent evaluation criteria across all datasets
    - Comprehensive model assessment
    """
    
    print("🏫 EMOTION CLASSIFICATION BENCHMARK SUITE")
    print("="*60)
    print("Testing model across multiple standardized emotion datasets...")
    print("Just like different exams test different aspects of knowledge,")
    print("different datasets test different aspects of emotion understanding.\n")
    
    # Initialize the framework (the "exam administrator")
    engine = BenchmarkEngine()
    
    # Register components (the "tools" students can use)
    engine.register_adapter("tflite", TensorFlowLiteAdapter)
    engine.register_adapter("huggingface", HuggingFaceAdapter)
    engine.register_metric("template_accuracy", TemplateAccuracyMetric)
    engine.register_metric("template_multilabel", TemplateMultiLabelMetric)
    engine.register_dataset("template", TemplateDataset)
    
    # Configure the benchmark (exam settings)
    engine.configure_benchmark({
        "num_samples": 500,  # Test on 500 samples per dataset
        "warmup_runs": 2,
        "batch_size": 1,
        "precision": "fp32",
        "device": "cpu"
    })
    
    # Test with TFLite model
    print("📝 STUDENT A: TensorFlow Lite Model")
    print("Model: notQuantizedModel.tflite")
    print("Adapter: TensorFlow Lite (mobile/edge optimized)")
    print("-" * 40)
    
    model_path = "models/notQuantizedModel.tflite"
    model_config = {
        "task_type": "emotion-detection",
        "max_length": 512
    }
    
    # Run multi-dataset benchmark for emotion classification
    results = engine.run_multi_dataset_benchmark(
        task_type=TaskType.EMOTION_CLASSIFICATION,
        model_path=model_path,
        adapter_name="tflite",
        model_config=model_config
    )
    
    # Display comprehensive results
    engine.print_multi_dataset_results(results)
    
    return results


def benchmark_sentiment_analysis():
    """
    Benchmark sentiment analysis models across multiple datasets.
    
    Demonstrates how the framework adapts to different tasks:
    - Different task type (sentiment vs emotion)
    - Different datasets (SST-2, IMDB)
    - Different evaluation criteria
    """
    
    print("\n" + "="*60)
    print("📊 SENTIMENT ANALYSIS BENCHMARK SUITE")
    print("="*60)
    print("Testing sentiment analysis across standardized datasets...")
    print("Different task = different standardized test suite!\n")
    
    # Initialize framework
    engine = BenchmarkEngine()
    
    # Register components
    engine.register_adapter("huggingface", HuggingFaceAdapter)
    engine.register_metric("template_accuracy", TemplateAccuracyMetric)
    engine.register_dataset("template", TemplateDataset)
    
    # Configure benchmark
    engine.configure_benchmark({
        "num_samples": 300,
        "warmup_runs": 2,
        "batch_size": 1,
        "precision": "fp32",
        "device": "cpu"
    })
    
    # Test with HuggingFace model
    print("📝 STUDENT B: HuggingFace Model")
    print("Model: distilbert-base-uncased-finetuned-sst-2-english")
    print("Adapter: HuggingFace (transformer-based)")
    print("-" * 40)
    
    model_path = "distilbert-base-uncased-finetuned-sst-2-english"
    model_config = {
        "task_type": "sentiment-analysis",
        "max_length": 512
    }
    
    # Run multi-dataset benchmark for sentiment analysis
    results = engine.run_multi_dataset_benchmark(
        task_type=TaskType.SENTIMENT_ANALYSIS,
        model_path=model_path,
        adapter_name="huggingface",
        model_config=model_config
    )
    
    # Display results
    engine.print_multi_dataset_results(results)
    
    return results


def demonstrate_framework_capabilities():
    """
    Demonstrate the framework's capabilities as a standardized testing system.
    """
    
    print("\n" + "🎯 FRAMEWORK CAPABILITIES DEMONSTRATION")
    print("="*60)
    
    # Initialize framework
    engine = BenchmarkEngine()
    
    print("1. 📚 STANDARDIZED TEST SUITES")
    print("   - Each task type has its own set of standardized datasets")
    print("   - Consistent evaluation criteria across all datasets")
    print("   - Expected accuracy ranges for validation")
    
    print("\n2. 🏫 EXAM ADMINISTRATOR ROLE")
    print("   - Framework provides the 'exam papers' (datasets)")
    print("   - Framework provides the 'scoring rubrics' (metrics)")
    print("   - Framework enforces the 'exam rules' (data contracts)")
    
    print("\n3. 📝 STUDENT FLEXIBILITY")
    print("   - Students bring their own 'knowledge' (models)")
    print("   - Students bring their own 'pencils' (adapters)")
    print("   - Framework translates everything to common language")
    
    print("\n4. 🔄 TASK-SPECIFIC TESTING")
    print("   - Emotion Classification: 3 different emotion datasets")
    print("   - Sentiment Analysis: 2 different sentiment datasets")
    print("   - Text Classification: 1 news classification dataset")
    
    print("\n5. ✅ VALIDATION & ASSESSMENT")
    print("   - Results validated against expected accuracy ranges")
    print("   - Overall assessment: PASS/FAIL based on performance")
    print("   - Comprehensive reporting across all datasets")
    
    # Show available task types
    print(f"\n6. 📋 SUPPORTED TASK TYPES:")
    for task_type in engine.dataset_registry.get_task_types():
        datasets = engine.dataset_registry.get_datasets_for_task(task_type)
        print(f"   - {task_type.value}: {len(datasets)} datasets")
    
    print("\n" + "="*60)


def main():
    """Main function demonstrating multi-dataset benchmarking."""
    
    print("🎓 BENCHMARK ENGINE: STANDARDIZED ML TESTING FRAMEWORK")
    print("="*60)
    print("This framework acts as an impartial exam administrator,")
    print("providing standardized test suites for different ML tasks.")
    print("Each task type has its own set of 'exams' (datasets)")
    print("and consistent evaluation criteria (metrics).")
    print("="*60)
    
    # Check if required files exist
    model_path = "models/notQuantizedModel.tflite"
    if not os.path.exists(model_path):
        print(f"⚠️  Model file not found: {model_path}")
        print("   Skipping TFLite benchmark...")
        return
    
    # Demonstrate framework capabilities
    demonstrate_framework_capabilities()
    
    # Run emotion classification benchmark
    try:
        emotion_results = benchmark_emotion_classification()
        print(f"\n✅ Emotion classification benchmark completed!")
    except Exception as e:
        print(f"\n❌ Emotion classification benchmark failed: {e}")
    
    # Run sentiment analysis benchmark (if HuggingFace available)
    try:
        sentiment_results = benchmark_sentiment_analysis()
        print(f"\n✅ Sentiment analysis benchmark completed!")
    except Exception as e:
        print(f"\n❌ Sentiment analysis benchmark failed: {e}")
        print("   (This is expected if HuggingFace models aren't available)")
    
    print("\n" + "="*60)
    print("🎯 FRAMEWORK SUMMARY")
    print("="*60)
    print("✅ Standardized test suites for different ML tasks")
    print("✅ Consistent evaluation across multiple datasets")
    print("✅ Fair comparison between different model types")
    print("✅ Validation against expected performance ranges")
    print("✅ Comprehensive reporting and assessment")
    print("="*60)


if __name__ == "__main__":
    main()
