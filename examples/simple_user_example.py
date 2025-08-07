"""
Simple User Example - Following User's Pseudocode Design

This shows how a user would implement their model using the simplified API.
The framework provides the template, user just fills in the specifics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.simple_interfaces import (
    TFLiteModelAdapter, 
    HuggingFaceModelAdapter,
    ModelKind,
    SimpleBenchmarkEngine
)


class DummyEmotionModel:
    """
    Simple dummy model that doesn't require external dependencies.
    Perfect for testing the framework!
    """
    
    def kind(self) -> ModelKind:
        return ModelKind.EMOTION_CLASSIFIER
    
    def process(self, txt: str) -> any:
        """Simple dummy processing - just return a dummy emotion."""
        # Simple rule-based emotion detection
        txt_lower = txt.lower()
        if any(word in txt_lower for word in ['love', 'amazing', 'fantastic', 'great', 'excellent']):
            return ['joy']
        elif any(word in txt_lower for word in ['hate', 'terrible', 'awful', 'disappointing']):
            return ['sadness']
        elif any(word in txt_lower for word in ['fear', 'scary', 'terrifying']):
            return ['fear']
        else:
            return ['neutral']
    
    def run(self, txt: str) -> any:
        """Simple run method that just calls process."""
        return self.process(txt)


class DummySentimentModel:
    """
    Simple dummy sentiment model.
    """
    
    def kind(self) -> ModelKind:
        return ModelKind.SENTIMENT_ANALYZER
    
    def process(self, txt: str) -> any:
        """Simple sentiment analysis."""
        txt_lower = txt.lower()
        if any(word in txt_lower for word in ['love', 'amazing', 'great', 'fantastic', 'excellent', 'outstanding']):
            return 'positive'
        elif any(word in txt_lower for word in ['hate', 'terrible', 'awful', 'disappointing', 'poor', 'mediocre']):
            return 'negative'
        else:
            return 'neutral'
    
    def run(self, txt: str) -> any:
        return self.process(txt)


class DummyTextClassifier:
    """
    Simple dummy text classifier.
    """
    
    def kind(self) -> ModelKind:
        return ModelKind.TEXT_CLASSIFIER
    
    def process(self, txt: str) -> any:
        """Simple text classification."""
        txt_lower = txt.lower()
        if any(word in txt_lower for word in ['stock', 'market', 'economy', 'tesla', 'apple', 'google', 'microsoft']):
            return 'business'
        elif any(word in txt_lower for word in ['united', 'liverpool', 'barcelona', 'championship', 'match', 'player']):
            return 'sports'
        elif any(word in txt_lower for word in ['iphone', 'ai', 'software', 'platform']):
            return 'technology'
        else:
            return 'world'
    
    def run(self, txt: str) -> any:
        return self.process(txt)


class MyAwesomeModel(TFLiteModelAdapter):
    """
    User's awesome model implementation following their pseudocode exactly.
    
    The framework provides the template, user just fills in the specifics.
    """
    
    def kind(self) -> ModelKind:
        """Return the kind/task type of this model."""
        return ModelKind.EMOTION_CLASSIFIER
    
    def model_location(self) -> str:
        """Return path to the TFLite model file."""
        return "./models/notQuantizedModel.tflite"
    
    # preprocess: nothing to do (uses default)
    
    # process: nothing to do (uses TFLite default)
    
    def postprocess(self, raw: any) -> str:
        """
        Postprocess raw model output to emotion labels.
        Following user's pseudocode exactly.
        """
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
        
        return "neutral"  # fallback


class MySentimentModel(HuggingFaceModelAdapter):
    """
    User's sentiment analysis model.
    """
    
    def kind(self) -> ModelKind:
        return ModelKind.SENTIMENT_ANALYZER
    
    def model_name(self) -> str:
        return "distilbert-base-uncased-finetuned-sst-2-english"
    
    def postprocess(self, raw: any) -> str:
        """Convert logits to sentiment."""
        import torch
        
        if isinstance(raw, torch.Tensor):
            probabilities = torch.softmax(raw, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1)
            
            # Map to sentiment labels
            sentiment_map = ["negative", "positive"]
            return sentiment_map[prediction.item()]
        
        return "neutral"  # fallback


class MyTextClassifier(HuggingFaceModelAdapter):
    """
    User's text classification model.
    """
    
    def kind(self) -> ModelKind:
        return ModelKind.TEXT_CLASSIFIER
    
    def model_name(self) -> str:
        return "textattack/bert-base-uncased-ag-news"
    
    def postprocess(self, raw: any) -> str:
        """Convert logits to news category."""
        import torch
        
        if isinstance(raw, torch.Tensor):
            probabilities = torch.softmax(raw, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1)
            
            # Map to news categories
            category_map = ["world", "sports", "business", "technology"]
            return category_map[prediction.item()]
        
        return "unknown"  # fallback


def main():
    """
    Main function following user's pseudocode exactly.
    
    As Framework - Running the benchmark:
    class Main:
        def models_under_consideration() -> ModelBaseAdapter[]:
            models = list()
            models.add(new MyAwesomeModel)
        
        def run():
            registry = new DatasetRegistry()
            for(model in models_under_considerations())
                k = model.kind()
                datasets = registry.get_compatible(k)
                for (ds in datasets)
                    evaluate(model, ds)
    """
    
    print("🎯 SIMPLIFIED BENCHMARK FRAMEWORK")
    print("="*50)
    print("Following user's pseudocode design exactly!")
    print("Framework provides template, user fills specifics.")
    print("="*50)
    
    # Create the benchmark engine
    engine = SimpleBenchmarkEngine()
    
    # Add models to consideration (following user's pseudocode)
    # Start with dummy models that don't require external dependencies
    engine.add_model(DummyEmotionModel())
    engine.add_model(DummySentimentModel())
    engine.add_model(DummyTextClassifier())
    
    # Try to add the real models (will fail gracefully if dependencies missing)
    try:
        engine.add_model(MyAwesomeModel())
    except Exception as e:
        print(f"⚠️  Could not add TFLite model: {e}")
    
    try:
        engine.add_model(MySentimentModel())
    except Exception as e:
        print(f"⚠️  Could not add HuggingFace sentiment model: {e}")
    
    try:
        engine.add_model(MyTextClassifier())
    except Exception as e:
        print(f"⚠️  Could not add HuggingFace text classifier: {e}")
    
    # Run the benchmark (following user's pseudocode)
    results = engine.run()
    
    # Print summary
    print("\n" + "="*50)
    print("📊 BENCHMARK SUMMARY")
    print("="*50)
    
    for result in results:
        model_name = result['model']
        dataset_name = result['dataset']
        accuracy = result['accuracy']
        expected_min, expected_max = result['expected_range']
        status = "✅ PASS" if expected_min <= accuracy <= expected_max else "❌ FAIL"
        
        print(f"{model_name} on {dataset_name}: {accuracy:.3f} {status}")
    
    print("="*50)
    print("🎉 Framework provides template, user fills specifics!")
    print("Much cleaner and more intuitive than complex APIs!")


if __name__ == "__main__":
    main()
