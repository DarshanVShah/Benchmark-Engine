"""
HuggingFace Emotion Classification Example

This example demonstrates how to use the BenchmarkEngine with a HuggingFace
emotion classification model. It shows the complete pipeline from model loading
to benchmarking results.
"""

import os
import sys

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import BenchmarkEngine
from plugins.huggingface_adapter import HuggingFaceAdapter
from benchmark_datasets.template_dataset import TemplateDataset
from metrics.template_metric import TemplateAccuracyMetric
from metrics.template_metric import TemplateMultiLabelMetric


def create_emotion_dataset():
    """Create a simple emotion dataset for testing."""
    import pandas as pd
    
    # Create directory if it doesn't exist
    os.makedirs("benchmark_datasets/localTestSets", exist_ok=True)
    
    # Sample emotion data
    data = {
        'text': [
            "I am so happy today!",
            "This makes me very angry.",
            "I feel sad about this.",
            "I'm excited about the news!",
            "This is disgusting.",
            "I love this so much!",
            "I'm afraid of what might happen.",
            "This brings me joy.",
            "I'm surprised by this.",
            "I trust this completely.",
            "I'm feeling optimistic about the future.",
            "This situation makes me nervous.",
            "I'm grateful for your help.",
            "This is disappointing.",
            "I'm feeling proud of my achievements."
        ],
        'emotion': [
            "joy",
            "anger", 
            "sadness",
            "excitement",
            "disgust",
            "love",
            "fear",
            "joy",
            "surprise",
            "trust",
            "optimism",
            "nervousness",
            "gratitude",
            "disappointment",
            "pride"
        ]
    }
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    output_path = "benchmark_datasets/localTestSets/emotion_test.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✓ Created emotion test dataset: {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Sample data:")
    print(df.head())
    
    return output_path


def test_huggingface_emotion_classifier():
    """Test HuggingFace emotion classifier with the framework."""
    print("🚀 HUGGINGFACE EMOTION CLASSIFICATION EXAMPLE")
    print("=" * 60)
    
    # Create test dataset
    dataset_path = create_emotion_dataset()
    
    # Dataset configuration
    dataset_config = {
        "file_format": "csv",
        "text_column": "text",
        "label_columns": ["emotion"],
        "task_type": "single-label",  # Single emotion per text
        "max_length": 128
    }
    
    print(f"\n📊 DATASET CONFIGURATION")
    print(f"  Path: {dataset_path}")
    print(f"  Format: {dataset_config['file_format']}")
    print(f"  Text column: {dataset_config['text_column']}")
    print(f"  Labels: {dataset_config['label_columns']}")
    print(f"  Task type: {dataset_config['task_type']}")
    
    # Create benchmark engine
    print(f"\n🔧 SETTING UP BENCHMARK ENGINE")
    engine = BenchmarkEngine()
    
    # Register components
    engine.register_adapter("huggingface", HuggingFaceAdapter)
    engine.register_dataset("template", TemplateDataset)
    
    # Model configuration for emotion classification
    model_config = {
        "model_name": "j-hartmann/emotion-english-distilroberta-base",  # Pre-trained emotion classifier
        "device": "cpu",  # Use CPU for compatibility
        "max_length": 128,
        "input_type": "text",
        "output_type": "class_id",  # Single emotion prediction
        "task_type": "single-label",
        "is_multi_label": False
    }
    
    print(f"✓ Registered HuggingFace adapter")
    print(f"✓ Registered template dataset")
    
    # Load model
    print(f"\n🤖 LOADING HUGGINGFACE MODEL")
    print(f"Model: {model_config['model_name']}")
    
    if not engine.load_model("huggingface", model_config["model_name"], model_config):
        print("❌ Failed to load HuggingFace model")
        print("Note: This requires transformers and torch to be installed")
        print("Install with: pip install transformers torch")
        return False
    
    print("✓ HuggingFace model loaded successfully")
    
    # Load dataset
    print(f"\n📚 LOADING DATASET")
    if not engine.load_dataset("template", dataset_path, dataset_config):
        print("❌ Failed to load dataset")
        return False
    
    print("✓ Dataset loaded successfully")
    
    # Add accuracy metric
    metric = TemplateAccuracyMetric(input_type="class_id")
    engine.add_metric("accuracy", metric)
    print("✓ Added accuracy metric")
    
    # Validate setup
    print(f"\n✅ VALIDATING SETUP")
    if not engine.validate_setup():
        print("❌ Setup validation failed")
        print("Check the compatibility between model, dataset, and metric")
        return False
    
    print("✓ Setup validation passed")
    print(f"  - Dataset outputs: {engine.dataset.output_type.value}")
    print(f"  - Model expects: {engine.model_adapter.input_type.value}")
    print(f"  - Model outputs: {engine.model_adapter.output_type.value}")
    print(f"  - Metric expects: {engine.metrics[0].expected_input_type.value}")
    
    # Run benchmark
    print(f"\n🏃 RUNNING BENCHMARK")
    print("Testing on all samples in the dataset...")
    
    try:
        results = engine.run_benchmark()
        
        if results and "metrics" in results:
            print(f"\n🎯 BENCHMARK RESULTS")
            print("=" * 40)
            
            # Extract accuracy
            accuracy = None
            for metric_name, metric_values in results["metrics"].items():
                if isinstance(metric_values, dict) and "accuracy" in metric_values:
                    accuracy = metric_values["accuracy"]
                    break
            
            if accuracy is not None:
                print(f"Accuracy: {accuracy:.4f}")
                
                # Performance metrics
                if "timing" in results:
                    timing = results["timing"]
                    print(f"Total time: {timing.get('total_time', 'N/A')}")
                    print(f"Average inference: {timing.get('avg_inference_time', 'N/A')}")
                    print(f"Throughput: {timing.get('throughput', 'N/A')} samples/s")
                
                print(f"\n✅ Benchmark completed successfully!")
                return True
            else:
                print("❌ No accuracy found in results")
                return False
        else:
            print("❌ No results obtained")
            return False
            
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False


def test_multi_label_emotion():
    """Test multi-label emotion classification (if supported by the model)."""
    print(f"\n🔄 TESTING MULTI-LABEL EMOTION CLASSIFICATION")
    print("=" * 60)
    
    # Create benchmark engine
    engine = BenchmarkEngine()
    engine.register_adapter("huggingface", HuggingFaceAdapter)
    engine.register_dataset("template", TemplateDataset)
    
    # Multi-label model configuration
    model_config = {
        "model_name": "j-hartmann/emotion-english-distilroberta-base",
        "device": "cpu",
        "max_length": 128,
        "input_type": "text",
        "output_type": "probabilities",
        "task_type": "multi-label",
        "is_multi_label": True
    }
    
    # Load model
    if not engine.load_model("huggingface", model_config["model_name"], model_config):
        print("❌ Failed to load multi-label model")
        return False
    
    # Load dataset
    dataset_path = "benchmark_datasets/localTestSets/emotion_test.csv"
    dataset_config = {
        "file_format": "csv",
        "text_column": "text",
        "label_columns": ["emotion"],
        "task_type": "multi-label",
        "max_length": 128
    }
    
    if not engine.load_dataset("template", dataset_path, dataset_config):
        print("❌ Failed to load dataset for multi-label test")
        return False
    
    # Add multi-label metric
    metric = TemplateMultiLabelMetric(metric_type="accuracy", threshold=0.5)
    engine.add_metric("multi_label_accuracy", metric)
    
    # Validate setup
    if not engine.validate_setup():
        print("❌ Multi-label setup validation failed")
        return False
    
    print("✓ Multi-label setup validation passed")
    
    # Run quick test
    try:
        results = engine.run_benchmark(num_samples=5)
        if results and "metrics" in results:
            print("✓ Multi-label test completed successfully!")
            return True
        else:
            print("❌ Multi-label test failed")
            return False
    except Exception as e:
        print(f"❌ Multi-label test error: {e}")
        return False


def main():
    """Main function to run the HuggingFace emotion classification example."""
    print("🚀 HUGGINGFACE EMOTION CLASSIFICATION EXAMPLE")
    print("=" * 60)
    print("This example demonstrates:")
    print("1. Loading a pre-trained HuggingFace emotion classifier")
    print("2. Creating and loading a custom emotion dataset")
    print("3. Running benchmarks with the framework")
    print("4. Testing both single-label and multi-label classification")
    
    try:
        # Test single-label emotion classification
        success = test_huggingface_emotion_classifier()
        
        if success:
            print(f"\n🎉 Single-label emotion classification test passed!")
            
            # Test multi-label if single-label succeeded
            multi_label_success = test_multi_label_emotion()
            if multi_label_success:
                print(f"🎉 Multi-label emotion classification test also passed!")
            else:
                print(f"⚠️  Multi-label test failed (this is normal for some models)")
        else:
            print(f"\n❌ Single-label emotion classification test failed")
            return False
        
        print(f"\n✅ All tests completed!")
        print(f"\nTo use your own HuggingFace models:")
        print(f"1. Change the model_name in model_config")
        print(f"2. Adjust the task_type and is_multi_label settings")
        print(f"3. Modify the dataset configuration as needed")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
