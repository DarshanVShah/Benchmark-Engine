"""
Dataset Registry Example

This example demonstrates:
1. Dataset registry with local and remote datasets
2. Automatic download of remote datasets
3. Proper handling of multi-label vs single-label tasks
4. Compatibility validation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import BenchmarkEngine
from core.dataset_registry import TaskType, DatasetRegistry
from plugins import HuggingFaceAdapter
from benchmark_datasets import TemplateDataset
from metrics.template_metric import TemplateMultiLabelMetric, TemplateAccuracyMetric


def demonstrate_dataset_registry():
    """Demonstrate the dataset registry functionality."""
    print("📊 DATASET REGISTRY DEMONSTRATION")
    print("="*60)
    
    # Create registry
    registry = DatasetRegistry()
    
    # Show available task types
    print("\n1. Available Task Types:")
    for task_type in registry.get_task_types():
        datasets = registry.get_datasets_for_task(task_type)
        print(f"   - {task_type.value}: {len(datasets)} datasets")
    
    # Demonstrate emotion classification datasets
    print("\n2. Emotion Classification Datasets:")
    emotion_datasets = registry.get_datasets_for_task(TaskType.EMOTION_CLASSIFICATION)
    for dataset in emotion_datasets:
        status = "LOCAL" if not dataset.is_remote else "REMOTE"
        print(f"   - {dataset.name}: {dataset.description} ({status})")
    
    return registry


def test_local_dataset(registry: DatasetRegistry):
    """Test the local emotion dataset."""
    print("\n" + "="*60)
    print("TESTING LOCAL DATASET")
    print("="*60)
    
    # Get local emotion dataset
    local_dataset = None
    for dataset in registry.get_datasets_for_task(TaskType.EMOTION_CLASSIFICATION):
        if not dataset.is_remote:
            local_dataset = dataset
            break
    
    if not local_dataset:
        print("❌ No local dataset found")
        return False
    
    print(f"Testing local dataset: {local_dataset.name}")
    
    # Create engine
    engine = BenchmarkEngine()
    engine.register_adapter("huggingface", HuggingFaceAdapter)
    engine.register_dataset("template", TemplateDataset)
    
    # Load model
    model_config = {
        "device": "cpu",
        "precision": "fp32",
        "max_length": 128,
        "is_multi_label": True,
        "task_type": "multi-label"
    }
    
    success = engine.load_model("huggingface", "distilbert-base-uncased", model_config)
    if not success:
        print("❌ Failed to load model")
        return False
    
    # Ensure dataset is available
    if not registry.ensure_dataset_available(local_dataset):
        print("❌ Dataset not available")
        return False
    
    # Load dataset
    success = engine.load_dataset("template", local_dataset.path, local_dataset.config)
    if not success:
        print("❌ Failed to load dataset")
        return False
    
    # Add appropriate metric
    if local_dataset.config.get("task_type") == "multi-label":
        metric = TemplateMultiLabelMetric(metric_type="accuracy", threshold=0.5)
    else:
        metric = TemplateAccuracyMetric(input_type="class_id")
    
    engine.metrics = [metric]
    
    # Validate setup
    success = engine.validate_setup()
    if not success:
        print("❌ Setup validation failed")
        return False
    
    print("✓ Local dataset test setup successful!")
    print(f"  - Dataset: {local_dataset.name}")
    print(f"  - Task type: {local_dataset.config.get('task_type')}")
    print(f"  - Model output: {engine.model_adapter.output_type.value}")
    print(f"  - Metric expects: {engine.metrics[0].expected_input_type.value}")
    
    return True


def test_remote_dataset_download(registry: DatasetRegistry):
    """Test downloading a remote dataset."""
    print("\n" + "="*60)
    print("TESTING REMOTE DATASET DOWNLOAD")
    print("="*60)
    
    # Find a remote dataset
    remote_dataset = None
    for dataset in registry.get_datasets_for_task(TaskType.EMOTION_CLASSIFICATION):
        if dataset.is_remote:
            remote_dataset = dataset
            break
    
    if not remote_dataset:
        print("❌ No remote dataset found")
        return False
    
    print(f"Testing remote dataset: {remote_dataset.name}")
    print(f"Download URL: {remote_dataset.download_url}")
    
    # Test download
    success = registry.ensure_dataset_available(remote_dataset)
    if success:
        print("✓ Remote dataset downloaded successfully!")
        return True
    else:
        print("❌ Failed to download remote dataset")
        return False


def demonstrate_multi_dataset_benchmark():
    """Demonstrate multi-dataset benchmarking."""
    print("\n" + "="*60)
    print("MULTI-DATASET BENCHMARK DEMONSTRATION")
    print("="*60)
    
    # Create engine
    engine = BenchmarkEngine()
    engine.register_adapter("huggingface", HuggingFaceAdapter)
    engine.register_dataset("template", TemplateDataset)
    
    # Configure for emotion classification
    model_config = {
        "device": "cpu",
        "precision": "fp32",
        "max_length": 128
    }
    
    # Run benchmark on emotion classification task
    print("Running emotion classification benchmark...")
    results = engine.run_multi_dataset_benchmark(
        task_type=TaskType.EMOTION_CLASSIFICATION,
        model_path="distilbert-base-uncased",
        adapter_name="huggingface",
        model_config=model_config,
        dataset_names=["2018-E-c-En-test-gold"]  # Only test local dataset
    )
    
    if results and "datasets" in results:
        print("\nBenchmark Results:")
        for dataset_name, dataset_results in results["datasets"].items():
            if "error" in dataset_results:
                print(f"  {dataset_name}: ❌ {dataset_results['error']}")
            else:
                print(f"  {dataset_name}: ✅ Completed")
        
        if "validation" in results:
            validation = results["validation"]
            print(f"\nValidation: {validation['overall_assessment']}")
    
    return results


def main():
    """Main demonstration function."""
    print("🚀 DATASET REGISTRY EXAMPLE")
    print("="*60)
    print("This example demonstrates:")
    print("1. Dataset registry with local and remote datasets")
    print("2. Automatic download of remote datasets")
    print("3. Proper handling of multi-label vs single-label tasks")
    print("4. Compatibility validation")
    
    # Demonstrate registry
    registry = demonstrate_dataset_registry()
    
    # Test local dataset
    local_success = test_local_dataset(registry)
    
    # Test remote dataset download (optional - can be slow)
    print("\nNote: Remote dataset download test is optional and can be slow.")
    print("Uncomment the next line to test remote downloads:")
    # remote_success = test_remote_dataset_download(registry)
    
    # Demonstrate multi-dataset benchmark
    benchmark_results = demonstrate_multi_dataset_benchmark()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Local dataset test: {'✅ PASS' if local_success else '❌ FAIL'}")
    print(f"Multi-dataset benchmark: {'✅ PASS' if benchmark_results else '❌ FAIL'}")
    
    if local_success and benchmark_results:
        print("\n🎉 All tests passed! Dataset registry is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")


if __name__ == "__main__":
    main()
