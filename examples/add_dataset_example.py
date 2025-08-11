"""
Example: Adding Your Own Dataset to the Registry

This script demonstrates how to add custom datasets to the BenchmarkEngine.
"""

import os
import sys

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dataset_registry import DatasetRegistry, TaskType, DatasetConfig
from core.interfaces import DataType


def add_custom_dataset():
    """Example of adding a custom dataset to the registry."""
    print("📚 ADDING CUSTOM DATASET EXAMPLE")
    print("=" * 50)
    
    # Create registry instance
    registry = DatasetRegistry()
    
    # Example 1: Add a sentiment analysis dataset
    sentiment_config = DatasetConfig(
        name="MySentimentDataset",
        path="benchmark_datasets/localTestSets/my_sentiment.csv",
        task_type=TaskType.SENTIMENT_ANALYSIS,
        config={
            "file_format": "csv",
            "text_column": "text",
            "label_columns": ["negative", "positive"],
            "task_type": "single-label",
            "max_length": 512
        },
        description="My custom sentiment analysis dataset",
        expected_accuracy_range=(0.75, 0.95),
        is_remote=False
    )
    
    # Add to registry
    registry.add_dataset(sentiment_config)
    print(f"✓ Added dataset: {sentiment_config.name}")
    
    # Example 2: Add an emotion dataset
    emotion_config = DatasetConfig(
        name="MyEmotionDataset",
        path="benchmark_datasets/localTestSets/my_emotion.tsv",
        task_type=TaskType.EMOTION_CLASSIFICATION,
        config={
            "file_format": "tsv",
            "text_column": "sentence",
            "label_columns": ["joy", "sadness", "anger", "fear"],
            "task_type": "multi-label",
            "max_length": 256
        },
        description="My custom emotion classification dataset",
        expected_accuracy_range=(0.70, 0.90),
        is_remote=False
    )
    
    # Add to registry
    registry.add_dataset(emotion_config)
    print(f"✓ Added dataset: {emotion_config.name}")
    
    # Show all available datasets
    print(f"\n📊 Available datasets by task type:")
    for task_type in registry.get_task_types():
        datasets = registry.get_datasets_for_task(task_type)
        print(f"\n{task_type.value}:")
        for dataset in datasets:
            status = "LOCAL" if not dataset.is_remote else "REMOTE"
            print(f"  - {dataset.name}: {dataset.description} ({status})")
    
    return registry


def create_dataset_file_example():
    """Example of creating a simple dataset file."""
    print("\n📝 CREATING DATASET FILE EXAMPLE")
    print("=" * 50)
    
    import pandas as pd
    
    # Create directory if it doesn't exist
    os.makedirs("benchmark_datasets/localTestSets", exist_ok=True)
    
    # Create sample sentiment data
    data = {
        'text': [
            "I love this product!",
            "This is terrible quality.",
            "Amazing experience!",
            "Very disappointed.",
            "Great service!"
        ],
        'label': [
            "positive",
            "negative", 
            "positive",
            "negative",
            "positive"
        ]
    }
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    output_path = "benchmark_datasets/localTestSets/example_sentiment.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✓ Created example dataset: {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Sample data:")
    print(df.head())
    
    return output_path


def main():
    """Main function demonstrating dataset management."""
    print("🚀 DATASET MANAGEMENT EXAMPLE")
    print("=" * 50)
    
    # Create example dataset file
    dataset_path = create_dataset_file_example()
    
    # Add datasets to registry
    registry = add_custom_dataset()
    
    # Show how to use the registry
    print(f"\n🔍 USING THE REGISTRY:")
    print("=" * 50)
    
    # Get datasets for a specific task
    sentiment_datasets = registry.get_datasets_for_task(TaskType.SENTIMENT_ANALYSIS)
    print(f"Sentiment datasets: {len(sentiment_datasets)}")
    
    # Get a specific dataset by name
    my_dataset = registry.get_dataset_by_name("MySentimentDataset")
    if my_dataset:
        print(f"Found dataset: {my_dataset.name}")
        print(f"  Path: {my_dataset.path}")
        print(f"  Task: {my_dataset.task_type.value}")
        print(f"  Expected accuracy: {my_dataset.expected_accuracy_range}")
    
    print(f"\n✅ Dataset management example completed!")
    print(f"\nTo use your datasets:")
    print(f"1. Place dataset files in benchmark_datasets/localTestSets/")
    print(f"2. Add them to the registry using registry.add_dataset()")
    print(f"3. Use them in your benchmarks!")


if __name__ == "__main__":
    main()
