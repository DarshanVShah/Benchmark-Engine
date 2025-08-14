"""
Emotion Standardization Example

This example demonstrates how to use the standardized emotion system to:
1. Convert emotions from different models to standardized format
2. Get human-readable emotion labels instead of numbers
3. Map between different emotion classification schemes
4. Analyze emotion outputs with confidence scores
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    standardized_emotions, 
    map_emotion_to_standard,
    convert_emotion_output,
    get_emotion_summary,
    get_emotion_analysis
)


def demonstrate_emotion_mapping():
    """Demonstrate mapping emotions from different sources to standardized format."""
    print("=== Emotion Mapping Examples ===\n")
    
    # Example 1: Map from 2018-E-c-En dataset emotions
    print("1. Mapping from 2018-E-c-En dataset:")
    emotions_2018 = ["joy", "anger", "love", "optimism", "unknown_emotion"]
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
    
    # Example 3: Map from common model outputs
    print("3. Mapping from common model outputs:")
    common_emotions = ["happiness", "joy", "sadness", "fear", "excitement"]
    for emotion in common_emotions:
        standardized, confidence = map_emotion_to_standard(emotion, "common_models")
        print(f"   '{emotion}' -> '{standardized}' (confidence: {confidence:.2f})")
    
    print()
    
    # Example 4: Auto-detection
    print("4. Auto-detection mapping:")
    mixed_emotions = ["joy", 5, "unknown_emotion", "happiness"]
    for emotion in mixed_emotions:
        standardized, confidence = map_emotion_to_standard(emotion, "auto")
        print(f"   {emotion} -> '{standardized}' (confidence: {confidence:.2f})")


def demonstrate_output_conversion():
    """Demonstrate converting model outputs to standardized emotions."""
    print("\n=== Model Output Conversion Examples ===\n")
    
    # Example 1: Probability outputs (single sample)
    print("1. Converting probability outputs (single sample):")
    probabilities = np.array([0.1, 0.05, 0.3, 0.2, 0.15, 0.1, 0.1])
    result = convert_emotion_output(probabilities, "probabilities", top_k=3)
    print(f"   Raw probabilities: {probabilities}")
    print(f"   Standardized result: {result['top_emotion']} (confidence: {result['confidence']:.3f})")
    print(f"   Top 3 emotions: {[e['emotion'] for e in result['standardized_emotions']]}")
    
    print()
    
    # Example 2: Logit outputs
    print("2. Converting logit outputs:")
    logits = np.array([2.0, 1.0, 3.0, 0.5, 1.5, 0.8, 1.2])
    result = convert_emotion_output(logits, "logits", top_k=3)
    print(f"   Raw logits: {logits}")
    print(f"   Standardized result: {result['top_emotion']} (confidence: {result['confidence']:.3f})")
    
    print()
    
    # Example 3: Multi-label outputs
    print("3. Converting multi-label outputs:")
    multi_label = np.array([0.8, 0.2, 0.9, 0.1, 0.3, 0.7, 0.4])
    result = convert_emotion_output(multi_label, "multi_label", top_k=3)
    print(f"   Raw multi-label: {multi_label}")
    print(f"   Emotions detected: {result['num_emotions_detected']}")
    print(f"   Top emotion: {result['top_emotion']} (confidence: {result['confidence']:.3f})")
    
    print()
    
    # Example 4: Label outputs
    print("4. Converting label outputs:")
    labels = ["joy", "happiness", "excitement"]
    result = convert_emotion_output(labels, "labels", top_k=3)
    print(f"   Raw labels: {labels}")
    print(f"   Standardized result: {result['top_emotion']} (confidence: {result['confidence']:.3f})")


def demonstrate_emotion_analysis():
    """Demonstrate emotion analysis capabilities."""
    print("\n=== Emotion Analysis Examples ===\n")
    
    # Example 1: Analyze a probability output
    print("1. Analyzing probability output:")
    probabilities = np.array([0.1, 0.05, 0.3, 0.2, 0.15, 0.1, 0.1])
    result = convert_emotion_output(probabilities, "probabilities")
    analysis = get_emotion_analysis(result)
    print(f"   Primary emotion: {analysis['primary_emotion']}")
    print(f"   Confidence: {analysis['confidence']:.3f}")
    print(f"   Categories: {analysis['emotion_categories']}")
    print(f"   Arousal level: {analysis['arousal_level']}")
    print(f"   Valence: {analysis['valence']}")
    
    print()
    
    # Example 2: Get human-readable summary
    print("2. Human-readable summary:")
    summary = get_emotion_summary(result)
    print(f"   {summary}")


def demonstrate_emotion_categories():
    """Demonstrate emotion categorization capabilities."""
    print("\n=== Emotion Categories ===\n")
    
    # Show all emotion categories
    print("Available emotion categories:")
    for category, emotions in standardized_emotions.EMOTION_CATEGORIES.items():
        print(f"   {category}: {', '.join(emotions[:5])}{'...' if len(emotions) > 5 else ''}")
    
    print()
    
    # Show emotions in specific categories
    print("Emotions in specific categories:")
    categories_to_show = ["positive", "negative", "high_arousal", "low_arousal"]
    for category in categories_to_show:
        emotions = standardized_emotions.get_emotions_in_category(category)
        print(f"   {category}: {', '.join(emotions[:8])}{'...' if len(emotions) > 8 else ''}")
    
    print()
    
    # Show category membership for specific emotions
    print("Category membership for specific emotions:")
    emotions_to_check = ["joy", "anger", "fear", "contentment", "excitement"]
    for emotion in emotions_to_check:
        categories = standardized_emotions.get_emotion_categories(emotion)
        print(f"   {emotion}: {', '.join(categories)}")


def demonstrate_emotion_schemes():
    """Demonstrate different emotion schemes and their mappings."""
    print("\n=== Emotion Scheme Mappings ===\n")
    
    # Show 2018-E-c-En mapping
    print("1. 2018-E-c-En dataset mapping:")
    for original, standardized in standardized_emotions.mapping_2018_ec_en.items():
        print(f"   '{original}' -> '{standardized}'")
    
    print()
    
    # Show GoEmotions mapping (first 10)
    print("2. GoEmotions dataset mapping (first 10):")
    count = 0
    for emotion_id, emotion_name in standardized_emotions.mapping_goemotions.items():
        if count < 10:
            print(f"   ID {emotion_id}: '{emotion_name}'")
            count += 1
        else:
            break
    print("   ... (and more)")
    
    print()
    
    # Show common model mapping
    print("3. Common model mapping:")
    for original, standardized in standardized_emotions.mapping_common_models.items():
        print(f"   '{original}' -> '{standardized}'")


def main():
    """Run all demonstration functions."""
    print("Emotion Standardization System Demonstration")
    print("=" * 50)
    
    # Run demonstrations
    demonstrate_emotion_mapping()
    demonstrate_output_conversion()
    demonstrate_emotion_analysis()
    demonstrate_emotion_categories()
    demonstrate_emotion_schemes()
    
    print("\n" + "=" * 50)
    print("Demonstration complete!")
    print(f"Total standardized emotions available: {len(standardized_emotions.get_all_emotions())}")
    print(f"Standardized emotion set: {', '.join(standardized_emotions.get_all_emotions())}")


if __name__ == "__main__":
    main()
