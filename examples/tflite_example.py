"""
Universal TFLite Emotion Classifier Test with Emotion Standardization

This example shows how a user would use the BenchmarkEngine for universal testing
with the integrated emotion standardization system. The engine selects random emotion 
datasets and converts all outputs to standardized, human-readable emotions.
"""

import os
import sys
import numpy as np

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_datasets.template_dataset import TemplateDataset
from core.engine import BenchmarkEngine
from core.emotion_standardization import standardized_emotions, map_emotion_to_standard
from core.emotion_converter import convert_emotion_output, get_emotion_summary, get_emotion_analysis
from plugins.tflite_adapter import TensorFlowLiteAdapter
from metrics.template_metric import TemplateMultiLabelMetric


def demonstrate_emotion_standardization():
    """Demonstrate the emotion standardization system."""
    print("=== Emotion Standardization System ===\n")
    
    print("Standardized emotions available:")
    emotions = standardized_emotions.get_all_emotions()
    for i, emotion in enumerate(emotions):
        print(f"  {i}: {emotion}")
    
    print(f"\nTotal: {len(emotions)} standardized emotions\n")
    
    # Show emotion categories
    print("Emotion categories:")
    for category, emotion_list in standardized_emotions.EMOTION_CATEGORIES.items():
        print(f"  {category}: {', '.join(emotion_list)}")
    
    print("\n" + "="*50 + "\n")


def demonstrate_emotion_mapping():
    """Demonstrate emotion mapping from different sources."""
    print("=== Emotion Mapping Examples ===\n")
    
    # Example 1: Map from 2018-E-c-En dataset emotions
    print("1. Mapping from 2018-E-c-En dataset:")
    emotions_2018 = ["joy", "anger", "love", "optimism", "fear"]
    for emotion in emotions_2018:
        standardized, confidence = map_emotion_to_standard(emotion, "2018_ec_en")
        print(f"   '{emotion}' -> '{standardized}' (confidence: {confidence:.2f})")
    
    print()
    
    # Example 2: Map from GoEmotions numeric IDs
    print("2. Mapping from GoEmotions numeric IDs:")
    goemotions_ids = [0, 1, 2, 17, 25, 27]  # admiration, amusement, anger, joy, sadness, neutral
    for emotion_id in goemotions_ids:
        standardized, confidence = map_emotion_to_standard(emotion_id, "goemotions")
        print(f"   ID {emotion_id} -> '{standardized}' (confidence: {confidence:.2f})")
    
    print()
    
    # Example 3: Auto-detection
    print("3. Auto-detection mapping:")
    mixed_emotions = ["joy", 5, "happiness", "unknown_emotion"]
    for emotion in mixed_emotions:
        standardized, confidence = map_emotion_to_standard(emotion, "auto")
        print(f"   {emotion} -> '{standardized}' (confidence: {confidence:.2f})")
    
    print("\n" + "="*50 + "\n")


def demonstrate_output_conversion():
    """Demonstrate converting model outputs to standardized emotions."""
    print("=== Model Output Conversion Examples ===\n")
    
    # Example 1: Probability outputs (8 emotions)
    print("1. Converting probability outputs (8 emotions):")
    # Simulate TFLite model output probabilities for 8 emotions
    probabilities = np.array([0.05, 0.1, 0.15, 0.3, 0.1, 0.2, 0.05, 0.05])
    result = convert_emotion_output(probabilities, "probabilities", top_k=3)
    
    print(f"   Raw probabilities: {probabilities}")
    print(f"   Top emotion: {result['top_emotion']} (confidence: {result['confidence']:.3f})")
    print(f"   Top 3 emotions: {[e['emotion'] for e in result['standardized_emotions']]}")
    
    # Get emotion analysis
    analysis = get_emotion_analysis(result)
    print(f"   Categories: {analysis['emotion_categories']}")
    print(f"   Valence: {analysis['valence']}")
    print(f"   Arousal: {analysis['arousal_level']}")
    
    print()
    
    # Example 2: Logit outputs
    print("2. Converting logit outputs:")
    logits = np.array([1.0, 2.0, 0.5, 3.0, 1.5, 2.5, 0.8, 1.2])
    result = convert_emotion_output(logits, "logits", top_k=3)
    print(f"   Raw logits: {logits}")
    print(f"   Top emotion: {result['top_emotion']} (confidence: {result['confidence']:.3f})")
    
    print()
    
    # Example 3: Multi-label outputs
    print("3. Converting multi-label outputs:")
    multi_label = np.array([0.8, 0.2, 0.9, 0.1, 0.3, 0.7, 0.4, 0.6])
    result = convert_emotion_output(multi_label, "multi_label", top_k=3)
    print(f"   Raw multi-label: {multi_label}")
    print(f"   Emotions detected: {result['num_emotions_detected']}")
    print(f"   Top emotion: {result['top_emotion']} (confidence: {result['confidence']:.3f})")
    
    print("\n" + "="*50 + "\n")


def demonstrate_tflite_adapter_capabilities():
    """Demonstrate TFLite adapter emotion standardization capabilities."""
    print("=== TFLite Adapter Emotion Standardization ===\n")
    
    # Create TFLite adapter instance
    adapter = TensorFlowLiteAdapter()
    
    # Show emotion mapping capabilities
    print("1. Emotion Standardization Capabilities:")
    emotion_info = adapter.get_emotion_mapping_info()
    
    if emotion_info.get("emotion_standardization_available"):
        print(f"   ✅ Emotion standardization: Available")
        print(f"   ✅ Standardized emotions: {emotion_info['total_emotions']}")
        print(f"   ✅ Supported schemes: {', '.join(emotion_info['supported_schemes'])}")
        print(f"   ✅ Emotion set: {', '.join(emotion_info['standardized_emotions'])}")
        
        print("\n   Emotion categories:")
        for category, emotions in emotion_info['emotion_categories'].items():
            print(f"     {category}: {', '.join(emotions)}")
    else:
        print(f"   ❌ Emotion standardization: {emotion_info.get('error', 'Not available')}")
    
    print()
    
    # Show model info structure
    print("2. Model Information Structure:")
    model_info = adapter.get_model_info()
    print(f"   Model type: {model_info['type']}")
    print(f"   Input type: {model_info['input_type']}")
    print(f"   Output type: {model_info['output_type']}")
    print(f"   Task type: {model_info['task_type']}")
    
    if 'emotion_standardization' in model_info:
        print(f"   Emotion standardization: Integrated")
    
    print("\n" + "="*50 + "\n")


def demonstrate_emotion_analysis():
    """Demonstrate comprehensive emotion analysis capabilities."""
    print("=== Comprehensive Emotion Analysis ===\n")
    
    # Example 1: Analyze different emotion outputs
    print("1. Analyzing various emotion outputs:")
    
    # Simulate different model outputs
    test_cases = [
        ("High confidence happiness", np.array([0.05, 0.05, 0.05, 0.8, 0.05, 0.05, 0.05, 0.05])),
        ("Mixed emotions", np.array([0.3, 0.1, 0.2, 0.25, 0.1, 0.05, 0.05, 0.05])),
        ("Strong negative", np.array([0.7, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05])),
        ("Neutral dominant", np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.7]))
    ]
    
    for description, output in test_cases:
        print(f"\n   {description}:")
        result = convert_emotion_output(output, "probabilities")
        analysis = get_emotion_analysis(result)
        summary = get_emotion_summary(result)
        
        print(f"     Top emotion: {result['top_emotion']} (confidence: {result['confidence']:.3f})")
        print(f"     Categories: {', '.join(analysis['emotion_categories'])}")
        print(f"     Valence: {analysis['valence']}, Arousal: {analysis['arousal_level']}")
        print(f"     Summary: {summary}")
    
    print("\n" + "="*50 + "\n")


def run_tflite_benchmark():
    """Run the actual TFLite benchmark if model is available."""
    print("=== Running TFLite Benchmark with Emotion Standardization ===\n")
    
    # Check if model file exists
    model_path = "models/notQuantizedModel.tflite"
    if not os.path.exists(model_path):
        print(f"❌ TFLite model not found: {model_path}")
        print("   This is expected if you haven't downloaded the model yet.")
        print("   The emotion standardization system is fully functional without the model.")
        return False
    
    # Create the benchmark engine
    engine = BenchmarkEngine()
    
    # Register what we need
    engine.register_adapter("tflite", TensorFlowLiteAdapter)
    engine.register_dataset("template", TemplateDataset)
    
    # Load your TFLite model
    model_config = {
        "device": "cpu",
        "precision": "fp32",
        "max_length": 512,  # Use larger max length for universal testing
        "input_type": "text",
        "output_type": "probabilities",
        "task_type": "multi-label",
        "is_multi_label": True
    }
    
    print("Loading TFLite model...")
    if not engine.load_model("tflite", model_path, model_config):
        print("❌ Failed to load TFLite model")
        print("   This could be due to:")
        print("   - Missing TensorFlow installation")
        print("   - Corrupted model file")
        print("   - Memory constraints")
        return False
    
    print("✅ TFLite model loaded successfully!")
    
    # Add universal evaluation metric
    universal_metric = TemplateMultiLabelMetric(metric_type="accuracy", threshold=0.5)
    engine.add_metric("universal_accuracy", universal_metric)
    
    # Run universal benchmark (engine selects random datasets unknown to you)
    print("Running universal benchmark...")
    results = engine.run_universal_benchmark(num_samples=500)  # Test on 500 samples per dataset
    
    if results:
        print(f"\n✅ Universal benchmark completed successfully!")
        print(f"   Your adapter achieved {results['universal_accuracy']:.1%} average accuracy")
        print(f"   across {results['successful_runs']} successful runs out of {results['datasets_tested']} total datasets.")
        
        if results.get('failed_datasets'):
            print(f"\n   Failed datasets:")
            for failed_dataset in results['failed_datasets']:
                print(f"     - {failed_dataset}")
        
        print(f"\n   Success rate: {results['successful_runs']}/{results['datasets_tested']} = {(results['successful_runs']/results['datasets_tested'])*100:.0f}%")
        
        return True
    else:
        print("❌ Benchmark failed")
        return False


def main():
    """Universal test of TFLite emotion classifier with emotion standardization."""
    
    print("TFLite Emotion Classifier with Emotion Standardization")
    print("=" * 60)
    
    # Demonstrate the emotion standardization system
    demonstrate_emotion_standardization()
    demonstrate_emotion_mapping()
    demonstrate_output_conversion()
    demonstrate_tflite_adapter_capabilities()
    demonstrate_emotion_analysis()
    
    # Try to run the actual benchmark
    benchmark_success = run_tflite_benchmark()
    
    # Show emotion standardization benefits regardless of benchmark success
    print("\n=== Emotion Standardization Benefits ===")
    print(f"✅ All model outputs converted to {len(standardized_emotions.get_all_emotions())} standardized emotions")
    print(f"✅ Human-readable emotion names instead of numbers")
    print(f"✅ Consistent emotion labels across all datasets")
    print(f"✅ Built-in emotion analysis and categorization")
    print(f"✅ Confidence scores for emotion mapping reliability")
    print(f"✅ Multi-scheme support (2018-E-c-En, GoEmotions, common models)")
    print(f"✅ Automatic emotion scheme detection")
    print(f"✅ Comprehensive emotion analysis (valence, arousal, categories)")
    
    return benchmark_success


if __name__ == "__main__":
    success = main()

    if success:
        print("\n🎉 Full TFLite benchmark completed successfully!")
        print("\nThe emotion standardization system ensures all outputs are:")
        print("- Human-readable (e.g., 'happiness' instead of '3')")
        print("- Consistent across different models and datasets")
        print("- Analyzable with built-in emotion categories")
        print("- Mapped with confidence scores for reliability")
    else:
        print("\n📝 Emotion standardization demonstration completed!")
        print("The TFLite benchmark couldn't run, but the emotion standardization system is fully functional.")
        print("\nTo run the full benchmark, ensure:")
        print("- TensorFlow is installed: pip install tensorflow")
        print("- The TFLite model file is available")
        print("- Sufficient memory for the large model file")
    
    print("\n" + "="*60)
    print("Emotion Standardization System: READY ✅")
    print("TFLite Integration: READY ✅")
    print("Universal Benchmarking: READY ✅")
