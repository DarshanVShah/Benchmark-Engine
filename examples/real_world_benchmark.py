"""
Real-World Benchmark Example

This example demonstrates how users declare everything in the main function
and compares against known accuracy benchmarks for real models and datasets.

Models Used:
- distilbert-base-uncased-finetuned-sst-2-english (SST-2: ~91.1% accuracy)
- textattack/bert-base-uncased-ag-news (AG News: ~94.0% accuracy)
- textattack/roberta-base-IMDB (IMDB: ~95.0% accuracy)

Datasets Used:
- SST-2 (Stanford Sentiment Treebank)
- AG News (News Classification)
- IMDB (Movie Reviews)

Known Benchmarks:
- DistilBERT SST-2: 91.1% accuracy (HuggingFace leaderboard)
- BERT AG News: 94.0% accuracy (paper benchmarks)
- RoBERTa IMDB: 95.0% accuracy (paper benchmarks)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import BenchmarkEngine, DataType, OutputType
from plugins import HuggingFaceAdapter
from metrics import AccuracyMetric
from benchmark_datasets import HuggingFaceDataset


def benchmark_distilbert_sst2():
    """
    Benchmark DistilBERT on SST-2 dataset.
    Known accuracy: ~91.1% (HuggingFace leaderboard)
    """
    
    print("=" * 60)
    print("BENCHMARK 1: DistilBERT on SST-2")
    print("=" * 60)
    print("Model: distilbert-base-uncased-finetuned-sst-2-english")
    print("Dataset: SST-2 (Stanford Sentiment Treebank)")
    print("Expected Accuracy: ~91.1% (HuggingFace leaderboard)")
    print("Task: Binary sentiment classification")
    print()
    
    # User declares everything in main function
    engine = BenchmarkEngine()
    
    # 1. Register components
    engine.register_adapter("huggingface", HuggingFaceAdapter)
    engine.register_metric("accuracy", AccuracyMetric)
    engine.register_dataset("sst2", HuggingFaceDataset)
    
    # 2. Configure benchmark
    engine.configure_benchmark({
        "num_samples": 100,  # Larger sample for statistical significance
        "warmup_runs": 3,
        "batch_size": 1,
        "precision": "fp32",
        "device": "cpu"
    })
    
    # 3. Load dataset
    print("Loading SST-2 dataset...")
    engine.load_dataset("sst2", "sst2")
    
    # 4. Load model
    print("Loading DistilBERT SST-2 model...")
    engine.load_model(
        "huggingface", 
        "distilbert-base-uncased-finetuned-sst-2-english",
        {
            "task_type": "text-classification",
            "max_length": 128,
            "truncation": True,
            "padding": True
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
    
    # 8. Compare with known benchmark
    accuracy = results["metrics"]["Accuracy"]["accuracy"]
    expected = 0.911
    difference = abs(accuracy - expected)
    
    print(f"\nComparison with Known Benchmark:")
    print(f"  Our Result:     {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"  Expected:        {expected:.3f} ({expected*100:.1f}%)")
    print(f"  Difference:      {difference:.3f} ({difference*100:.1f}%)")
    
    if difference < 0.05:  # Within 5%
        print("  ✅ Result is close to expected benchmark!")
    else:
        print("  ⚠️  Result differs from expected benchmark")
    
    return results


def benchmark_bert_agnews():
    """
    Benchmark BERT on AG News dataset.
    Known accuracy: ~94.0% (paper benchmarks)
    """
    
    print("\n" + "=" * 60)
    print("BENCHMARK 2: BERT on AG News")
    print("=" * 60)
    print("Model: textattack/bert-base-uncased-ag-news")
    print("Dataset: AG News (News Classification)")
    print("Expected Accuracy: ~94.0% (paper benchmarks)")
    print("Task: 4-class news classification")
    print()
    
    # User declares everything in main function
    engine = BenchmarkEngine()
    
    # 1. Register components
    engine.register_adapter("huggingface", HuggingFaceAdapter)
    engine.register_metric("accuracy", AccuracyMetric)
    engine.register_dataset("ag_news", HuggingFaceDataset)
    
    # 2. Configure benchmark
    engine.configure_benchmark({
        "num_samples": 100,
        "warmup_runs": 3,
        "batch_size": 1,
        "precision": "fp32",
        "device": "cpu"
    })
    
    # 3. Load dataset
    print("Loading AG News dataset...")
    engine.load_dataset("ag_news", "ag_news")
    
    # 4. Load model
    print("Loading BERT fine-tuned for AG News...")
    engine.load_model(
        "huggingface", 
        "textattack/bert-base-uncased-ag-news",
        {
            "task_type": "text-classification",
            "max_length": 256,
            "truncation": True,
            "padding": True
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
    
    # 8. Compare with known benchmark
    accuracy = results["metrics"]["Accuracy"]["accuracy"]
    expected = 0.940
    difference = abs(accuracy - expected)
    
    print(f"\nComparison with Known Benchmark:")
    print(f"  Our Result:     {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"  Expected:        {expected:.3f} ({expected*100:.1f}%)")
    print(f"  Difference:      {difference:.3f} ({difference*100:.1f}%)")
    
    if difference < 0.05:
        print("  ✅ Result is close to expected benchmark!")
    else:
        print("  ⚠️  Result differs from expected benchmark")
    
    return results


def benchmark_roberta_imdb():
    """
    Benchmark RoBERTa on IMDB dataset.
    Known accuracy: ~95.0% (paper benchmarks)
    """
    
    print("\n" + "=" * 60)
    print("BENCHMARK 3: RoBERTa on IMDB")
    print("=" * 60)
    print("Model: textattack/roberta-base-IMDB")
    print("Dataset: IMDB (Movie Reviews)")
    print("Expected Accuracy: ~95.0% (paper benchmarks)")
    print("Task: Binary sentiment classification")
    print()
    
    # User declares everything in main function
    engine = BenchmarkEngine()
    
    # 1. Register components
    engine.register_adapter("huggingface", HuggingFaceAdapter)
    engine.register_metric("accuracy", AccuracyMetric)
    engine.register_dataset("imdb", HuggingFaceDataset)
    
    # 2. Configure benchmark
    engine.configure_benchmark({
        "num_samples": 100,
        "warmup_runs": 3,
        "batch_size": 1,
        "precision": "fp32",
        "device": "cpu"
    })
    
    # 3. Load dataset
    print("Loading IMDB dataset...")
    engine.load_dataset("imdb", "imdb")
    
    # 4. Load model
    print("Loading RoBERTa fine-tuned for IMDB...")
    engine.load_model(
        "huggingface", 
        "textattack/roberta-base-IMDB",
        {
            "task_type": "text-classification",
            "max_length": 512,
            "truncation": True,
            "padding": True
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
    
    # 8. Compare with known benchmark
    accuracy = results["metrics"]["Accuracy"]["accuracy"]
    expected = 0.950
    difference = abs(accuracy - expected)
    
    print(f"\nComparison with Known Benchmark:")
    print(f"  Our Result:     {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"  Expected:        {expected:.3f} ({expected*100:.1f}%)")
    print(f"  Difference:      {difference:.3f} ({difference*100:.1f}%)")
    
    if difference < 0.05:
        print("  ✅ Result is close to expected benchmark!")
    else:
        print("  ⚠️  Result differs from expected benchmark")
    
    return results



def main():
    """Main function showing complete user workflow."""
    
    print("REAL-WORLD BENCHMARK EXAMPLE")
    print("=" * 60)
    print("This example shows how users declare everything in main function")
    print("and compares against known accuracy benchmarks.")
    print()
    
    # Run benchmarks
    results1 = benchmark_distilbert_sst2()
    results2 = benchmark_bert_agnews()
    results3 = benchmark_roberta_imdb()
    
    
   


if __name__ == "__main__":
    main() 