"""
Simple TFLite Emotion Classifier Test

This example tests the TFLite emotion classifier model using the improved framework.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.simple_interfaces import TFLiteModelAdapter, ModelKind, SimpleBenchmarkEngine


class EmotionClassifier(TFLiteModelAdapter):
    """
    Simple TFLite emotion classifier implementation.
    """
    
    def kind(self) -> ModelKind:
        return ModelKind.EMOTION_CLASSIFIER
    
    def model_location(self) -> str:
        return "./models/notQuantizedModel.tflite"
    
    def postprocess(self, raw: any) -> any:
        """Convert model output to emotion labels."""
        import numpy as np
        
        # Convert raw logits to probabilities
        if isinstance(raw, np.ndarray):
            # Apply sigmoid for multi-label
            probabilities = 1 / (1 + np.exp(-raw))
            
            # Map to emotion labels based on threshold
            threshold = 0.5
            emotions = []
            
            # Map indices to emotion names
            emotion_map = [
                "anger", "anticipation", "disgust", "fear", "joy",
                "love", "optimism", "pessimism", "sadness", "surprise", "trust"
            ]
            
            for i, prob in enumerate(probabilities[0]):
                if prob > threshold and i < len(emotion_map):
                    emotions.append(emotion_map[i])
            
            return emotions if emotions else ["neutral"]
        
        return ["neutral"]  # fallback


def main():
    """Test the TFLite emotion classifier."""
    
    # Create benchmark engine
    engine = SimpleBenchmarkEngine()
    
    # Add the emotion classifier
    engine.add_model(EmotionClassifier())
    
    # Run the benchmark
    results = engine.run()    
    
    for result in results:
        model_name = result['model']
        dataset_name = result['dataset']
        accuracy = result['accuracy']
        expected_min, expected_max = result['expected_range']
        status = "PASS" if expected_min <= accuracy <= expected_max else "FAIL"
        
        print(f"{model_name} on {dataset_name}: {accuracy:.3f} {status}")
    

if __name__ == "__main__":
    main()
