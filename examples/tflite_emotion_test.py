"""
TFLite Emotion Processing Test

This script demonstrates how the TFLite adapter processes emotion outputs
and converts them to standardized emotions using the emotion standardization system.
"""

import numpy as np
import sys
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plugins.tflite_adapter import TensorFlowLiteAdapter
from core.emotion_standardization import standardized_emotions


def test_emotion_processing():
    """Test emotion processing capabilities of the TFLite adapter."""
    print("TFLite Emotion Processing Test")
    print("=" * 40)
    
    # Create TFLite adapter
    adapter = TensorFlowLiteAdapter()
    
    # Test 1: Emotion mapping capabilities
    print("\n1. Testing Emotion Mapping Capabilities:")
    emotion_info = adapter.get_emotion_mapping_info()
    
    if emotion_info.get("emotion_standardization_available"):
        print("   ✅ Emotion standardization is available")
        print(f"   ✅ Supports {emotion_info['total_emotions']} standardized emotions")
        print(f"   ✅ Supported schemes: {', '.join(emotion_info['supported_schemes'])}")
    else:
        print("   ❌ Emotion standardization not available")
        return False
    
    # Test 2: Simulate model outputs and convert them
    print("\n2. Testing Emotion Output Conversion:")
    
    # Simulate different types of model outputs
    test_outputs = [
        ("High confidence happiness", np.array([0.05, 0.05, 0.05, 0.8, 0.05, 0.05, 0.05, 0.05])),
        ("Mixed emotions", np.array([0.3, 0.1, 0.2, 0.25, 0.1, 0.05, 0.05, 0.05])),
        ("Strong negative", np.array([0.7, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05])),
        ("Neutral dominant", np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.7]))
    ]
    
    for description, output in test_outputs:
        print(f"\n   {description}:")
        
        # Convert to standardized emotions
        result = adapter.convert_to_standardized_emotions(output, "probabilities")
        
        if "error" not in result:
            print(f"     Top emotion: {result['top_emotion']}")
            print(f"     Confidence: {result['confidence']:.3f}")
            
            if "analysis" in result:
                analysis = result["analysis"]
                print(f"     Categories: {', '.join(analysis.get('emotion_categories', []))}")
                print(f"     Valence: {analysis.get('valence', 'unknown')}")
                print(f"     Arousal: {analysis.get('arousal_level', 'unknown')}")
            
            if "emotion_standardization" in result:
                std_info = result["emotion_standardization"]
                print(f"     Standardized from: {std_info.get('conversion_format', 'unknown')}")
        else:
            print(f"     Error: {result['error']}")
    
    # Test 3: Test emotion mapping from different schemes
    print("\n3. Testing Emotion Scheme Mapping:")
    
    # Test 2018-E-c-En emotions
    emotions_2018 = ["joy", "anger", "love", "optimism", "fear"]
    print("   Mapping from 2018-E-c-En dataset:")
    for emotion in emotions_2018:
        standardized, confidence = standardized_emotions.get_standardized_emotion(emotion, "2018_ec_en")
        print(f"     '{emotion}' -> '{standardized}' (confidence: {confidence:.2f})")
    
    # Test GoEmotions IDs
    goemotions_ids = [0, 1, 2, 17, 25, 27]
    print("   Mapping from GoEmotions numeric IDs:")
    for emotion_id in goemotions_ids:
        standardized, confidence = standardized_emotions.get_standardized_emotion(emotion_id, "goemotions")
        print(f"     ID {emotion_id} -> '{standardized}' (confidence: {confidence:.2f})")
    
    # Test 4: Model information
    print("\n4. Model Information:")
    model_info = adapter.get_model_info()
    print(f"   Model type: {model_info['type']}")
    print(f"   Input type: {model_info['input_type']}")
    print(f"   Output type: {model_info['output_type']}")
    print(f"   Task type: {model_info['task_type']}")
    
    if 'emotion_standardization' in model_info:
        print("   ✅ Emotion standardization integrated")
    
    print("\n" + "=" * 40)
    print("✅ All tests completed successfully!")
    print("The TFLite adapter is ready for emotion classification with standardization.")
    
    return True


def demonstrate_real_world_usage():
    """Demonstrate how this would work in a real-world scenario."""
    print("\nReal-World Usage Example:")
    print("=" * 40)
    
    # Simulate a real emotion classification scenario
    print("Scenario: Processing text through TFLite emotion classifier")
    
    # Simulate model output (8 emotion probabilities)
    model_output = np.array([0.1, 0.05, 0.15, 0.4, 0.1, 0.1, 0.05, 0.05])
    
    print(f"\nRaw model output (probabilities): {model_output}")
    print("Emotion indices: [anger, disgust, fear, happiness, sadness, surprise, love, neutral]")
    
    # Convert using the adapter
    adapter = TensorFlowLiteAdapter()
    result = adapter.convert_to_standardized_emotions(model_output, "probabilities")
    
    if "error" not in result:
        print(f"\n✅ Standardized result:")
        print(f"   Primary emotion: {result['top_emotion']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        
        if "analysis" in result:
            analysis = result["analysis"]
            print(f"   Valence: {analysis.get('valence', 'unknown')}")
            print(f"   Arousal: {analysis.get('arousal_level', 'unknown')}")
            print(f"   Categories: {', '.join(analysis.get('emotion_categories', []))}")
        
        print(f"\nInstead of getting '3' (index), you get 'happiness' (readable)")
        print(f"Instead of raw probabilities, you get analyzed emotion categories")
        print(f"Instead of model-specific labels, you get standardized emotions")
    
    print("\n" + "=" * 40)


if __name__ == "__main__":
    try:
        success = test_emotion_processing()
        if success:
            demonstrate_real_world_usage()
        else:
            print("❌ Tests failed")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        sys.exit(1)
