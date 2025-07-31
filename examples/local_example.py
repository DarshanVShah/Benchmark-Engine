"""
DistilBERT SST-2 Benchmark Example

This example benchmarks the DistilBERT model fine-tuned on SST-2 (Stanford Sentiment Treebank)
for sentiment analysis. The model achieves 91.1% accuracy on the dev set.

Model: distilbert-base-uncased-finetuned-sst-2-english
Dataset: SST-2 (Stanford Sentiment Treebank)
Task: Binary sentiment classification (positive/negative)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import BenchmarkEngine
from plugins import HuggingFaceAdapter
from metrics import AccuracyMetric, ClassificationMetric
from benchmark_datasets import HuggingFaceDataset


def main():
    """Benchmark DistilBERT SST-2 model for sentiment analysis."""
    
    print("DistilBERT SST-2 Sentiment Analysis Benchmark")
    
    # Create the benchmark engine
    engine = BenchmarkEngine()
    
    # Register HuggingFace components
    engine.register_adapter("huggingface", HuggingFaceAdapter)
    engine.register_metric("accuracy", AccuracyMetric)
    engine.register_dataset("sst2", HuggingFaceDataset)
    
    # Configure benchmark parameters
    engine.configure_benchmark({
        "num_samples": 1000,  # Larger sample for better accuracy
        "warmup_runs": 3,    # Fewer warmup runs
        "batch_size": 1,      # Single sample inference
        "precision": "fp32",  # Full precision
        "device": "cpu"       # Use CPU for demo
    })
    
    # Load SST-2 dataset with proper configuration
    try:
        # Load SST-2 dataset
        engine.load_dataset("sst2", "sst2")
        
        # Verify we're getting actual sentences
        test_samples = engine.dataset.get_samples_with_targets(3)
        for i, (sample, target) in enumerate(test_samples):
            text = sample['text']
            if len(text) < 10 or text in ['idx', 'sentence', 'label']:
                print(f"Warning: Sample {i+1} has suspicious text: '{text}'")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Add accuracy metric
    engine.add_metric("accuracy")
    
    # Load DistilBERT SST-2 model
    engine.load_model(
        "huggingface", 
        "distilbert-base-uncased-finetuned-sst-2-english",
        {
            "task_type": "text-classification",
            "max_length": 128,  # SST-2 typical length
            "truncation": True,
            "padding": True
        }
    )
    
    # Run benchmark
    results = engine.run_benchmark()
    
    # Display results
    engine.print_results()
    
    # Export results
    engine.export_results("distilbert_sst2_results.json", format="json")
    engine.export_results("distilbert_sst2_results.md", format="markdown")
    

def demo_simple_test():
    """Simple test with dummy data to verify the model works."""
    
    print("\n" + "=" * 60)
    print("Simple Model Test")
    print("=" * 60)
    
    engine = BenchmarkEngine()
    engine.register_adapter("huggingface", HuggingFaceAdapter)
    
    # Load model
    print("Loading DistilBERT SST-2 model...")
    engine.load_model(
        "huggingface", 
        "distilbert-base-uncased-finetuned-sst-2-english",
        {"task_type": "text-classification"}
    )
    
    # Test sentences
    test_sentences = [
        "This movie is absolutely fantastic!",
        "I really enjoyed this film.",
        "This was the worst movie I've ever seen.",
        "The acting was terrible and the plot was boring.",
        "A masterpiece of cinema.",
        "Complete waste of time."
    ]
    
    print("\nTesting sample sentences:")
    print("-" * 40)
    
    for sentence in test_sentences:
        # Create a simple prediction
        try:
            # Use the correct method to run inference
            preprocessed = engine.model_adapter.preprocess_input({"text": sentence})
            model_output = engine.model_adapter.run(preprocessed)
            prediction = engine.model_adapter.postprocess_output(model_output)
            
            # Extract sentiment
            if isinstance(prediction[0], dict):
                sentiment = prediction[0].get("sentiment", "unknown")
                confidence = prediction[0].get("confidence", 0.0)
            else:
                sentiment = "positive" if prediction[0] == 1 else "negative"
                confidence = 0.5
            
            print(f"'{sentence}'")
            print(f"  Sentiment: {sentiment}")
            print(f"  Confidence: {confidence:.3f}")
            print()
        except Exception as e:
            print(f"Error processing '{sentence}': {e}")


if __name__ == "__main__":
    main()
    #demo_simple_test()
    
    
    print("DistilBERT SST-2 Benchmark Complete!")
    
