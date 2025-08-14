"""
Emotion Output Converter for BenchmarkEngine

This module provides utilities to convert raw model outputs (probabilities, logits, etc.)
into standardized emotion labels with confidence scores and human-readable results.
"""

from typing import Dict, List, Tuple, Union, Any
import numpy as np
from .emotion_standardization import standardized_emotions, map_emotion_to_standard


class EmotionOutputConverter:
    """
    Converts raw model outputs to standardized emotion labels.
    
    This class handles various output formats from different emotion classification models
    and converts them to our standardized emotion system with confidence scores.
    """
    
    def __init__(self):
        """Initialize the emotion output converter."""
        self.standardized_emotions = standardized_emotions
    
    def convert_model_output(self, 
                           model_output: Union[np.ndarray, List, Dict, Any],
                           output_format: str = "auto",
                           top_k: int = 3) -> Dict[str, Any]:
        """
        Convert model output to standardized emotion format.
        
        Args:
            model_output: Raw output from the model (probabilities, logits, etc.)
            output_format: Format of the output ("probabilities", "logits", "labels", "auto")
            top_k: Number of top emotions to return
        
        Returns:
            Dictionary with standardized emotion results
        """
        if output_format == "auto":
            output_format = self._detect_output_format(model_output)
        
        try:
            if output_format == "probabilities":
                return self._convert_probabilities(model_output, top_k)
            elif output_format == "logits":
                return self._convert_logits(model_output, top_k)
            elif output_format == "labels":
                return self._convert_labels(model_output, top_k)
            elif output_format == "multi_label":
                return self._convert_multi_label(model_output, top_k)
            else:
                return self._convert_generic(model_output, top_k)
        except Exception as e:
            return {
                "error": f"Failed to convert output: {str(e)}",
                "standardized_emotions": [],
                "top_emotion": "neutral",
                "confidence": 0.0
            }
    
    def _detect_output_format(self, model_output: Any) -> str:
        """Automatically detect the format of model output."""
        if isinstance(model_output, np.ndarray):
            if len(model_output.shape) == 1:
                if np.all((model_output >= 0) & (model_output <= 1)):
                    return "probabilities"
                else:
                    return "logits"
            elif len(model_output.shape) == 2:
                if np.all((model_output >= 0) & (model_output <= 1)):
                    return "probabilities"
                else:
                    return "logits"
        elif isinstance(model_output, list):
            if all(isinstance(x, (int, float)) for x in model_output):
                if all(0 <= x <= 1 for x in model_output):
                    return "probabilities"
                else:
                    return "logits"
            elif all(isinstance(x, str) for x in model_output):
                return "labels"
        elif isinstance(model_output, dict):
            if "probabilities" in model_output:
                return "probabilities"
            elif "logits" in model_output:
                return "logits"
            elif "labels" in model_output:
                return "labels"
        
        return "generic"
    
    def _convert_probabilities(self, 
                              probabilities: Union[np.ndarray, List], 
                              top_k: int) -> Dict[str, Any]:
        """Convert probability outputs to standardized emotions."""
        if isinstance(probabilities, list):
            probabilities = np.array(probabilities)
        
        # Get top-k emotions
        if len(probabilities.shape) == 1:
            # Single sample
            top_indices = np.argsort(probabilities)[-top_k:][::-1]
            top_emotions = []
            
            for idx in top_indices:
                emotion_name = self.standardized_emotions.get_emotion_name(idx)
                if emotion_name:
                    top_emotions.append({
                        "emotion": emotion_name,
                        "confidence": float(probabilities[idx]),
                        "original_index": int(idx)
                    })
            
            return {
                "standardized_emotions": top_emotions,
                "top_emotion": top_emotions[0]["emotion"] if top_emotions else "neutral",
                "confidence": top_emotions[0]["confidence"] if top_emotions else 0.0,
                "all_probabilities": probabilities.tolist()
            }
        else:
            # Multiple samples
            results = []
            for i in range(probabilities.shape[0]):
                sample_result = self._convert_probabilities(probabilities[i], top_k)
                results.append(sample_result)
            
            return {
                "samples": results,
                "batch_size": probabilities.shape[0]
            }
    
    def _convert_logits(self, 
                        logits: Union[np.ndarray, List], 
                        top_k: int) -> Dict[str, Any]:
        """Convert logit outputs to standardized emotions."""
        # Convert logits to probabilities
        if isinstance(logits, list):
            logits = np.array(logits)
        
        # Apply softmax to convert logits to probabilities
        probabilities = self._softmax(logits)
        
        # Use the probability conversion logic
        return self._convert_probabilities(probabilities, top_k)
    
    def _convert_labels(self, 
                        labels: Union[List, str], 
                        top_k: int) -> Dict[str, Any]:
        """Convert label outputs to standardized emotions."""
        if isinstance(labels, str):
            labels = [labels]
        
        standardized_results = []
        for label in labels:
            emotion, confidence = map_emotion_to_standard(label)
            standardized_results.append({
                "emotion": emotion,
                "confidence": confidence,
                "original_label": label
            })
        
        # Sort by confidence and take top-k
        standardized_results.sort(key=lambda x: x["confidence"], reverse=True)
        top_results = standardized_results[:top_k]
        
        return {
            "standardized_emotions": top_results,
            "top_emotion": top_results[0]["emotion"] if top_results else "neutral",
            "confidence": top_results[0]["confidence"] if top_results else 0.0,
            "all_labels": labels
        }
    
    def _convert_multi_label(self, 
                             multi_label_output: Union[np.ndarray, List], 
                             top_k: int) -> Dict[str, Any]:
        """Convert multi-label outputs to standardized emotions."""
        if isinstance(multi_label_output, list):
            multi_label_output = np.array(multi_label_output)
        
        # For multi-label, we get emotions where probability > threshold
        threshold = 0.5
        emotion_indices = np.where(multi_label_output > threshold)[0]
        
        top_emotions = []
        for idx in emotion_indices:
            emotion_name = self.standardized_emotions.get_emotion_name(idx)
            if emotion_name:
                top_emotions.append({
                    "emotion": emotion_name,
                    "confidence": float(multi_label_output[idx]),
                    "original_index": int(idx)
                })
        
        # Sort by confidence
        top_emotions.sort(key=lambda x: x["confidence"], reverse=True)
        top_emotions = top_emotions[:top_k]
        
        return {
            "standardized_emotions": top_emotions,
            "top_emotion": top_emotions[0]["emotion"] if top_emotions else "neutral",
            "confidence": top_emotions[0]["confidence"] if top_emotions else 0.0,
            "num_emotions_detected": len(emotion_indices),
            "all_probabilities": multi_label_output.tolist()
        }
    
    def _convert_generic(self, 
                         model_output: Any, 
                         top_k: int) -> Dict[str, Any]:
        """Convert generic model output to standardized emotions."""
        try:
            # Try to convert to numpy array
            if hasattr(model_output, 'numpy'):
                # TensorFlow/PyTorch tensor
                output_array = model_output.numpy()
            elif hasattr(model_output, 'tolist'):
                # NumPy array
                output_array = model_output
            else:
                # Try to convert to list
                output_array = np.array(model_output)
            
            # Try to detect format and convert
            return self.convert_model_output(output_array, "auto", top_k)
        except Exception as e:
            return {
                "error": f"Could not convert generic output: {str(e)}",
                "standardized_emotions": [],
                "top_emotion": "neutral",
                "confidence": 0.0
            }
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax function to convert logits to probabilities."""
        # Numerical stability: subtract max before exp
        x_stable = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_stable)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def get_emotion_summary(self, 
                           conversion_result: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of emotion conversion results.
        
        Args:
            conversion_result: Result from convert_model_output
        
        Returns:
            Human-readable summary string
        """
        if "error" in conversion_result:
            return f"Error: {conversion_result['error']}"
        
        if "samples" in conversion_result:
            # Batch processing
            return f"Processed {conversion_result['batch_size']} samples"
        
        top_emotion = conversion_result.get("top_emotion", "neutral")
        confidence = conversion_result.get("confidence", 0.0)
        
        summary = f"Primary emotion: {top_emotion} (confidence: {confidence:.3f})"
        
        if "standardized_emotions" in conversion_result:
            emotions = conversion_result["standardized_emotions"]
            if len(emotions) > 1:
                other_emotions = [e["emotion"] for e in emotions[1:3]]  # Top 2-3
                summary += f"\nOther detected emotions: {', '.join(other_emotions)}"
        
        return summary
    
    def get_emotion_analysis(self, 
                            conversion_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed analysis of emotion conversion results.
        
        Args:
            conversion_result: Result from convert_model_output
        
        Returns:
            Dictionary with analysis information
        """
        if "error" in conversion_result:
            return {"error": conversion_result["error"]}
        
        analysis = {
            "primary_emotion": conversion_result.get("top_emotion", "neutral"),
            "confidence": conversion_result.get("confidence", 0.0),
            "emotion_categories": [],
            "arousal_level": "unknown",
            "valence": "unknown"
        }
        
        top_emotion = conversion_result.get("top_emotion", "neutral")
        if top_emotion != "neutral":
            # Get emotion categories
            categories = self.standardized_emotions.get_emotion_categories(top_emotion)
            analysis["emotion_categories"] = categories
            
            # Determine arousal and valence
            if "high_arousal" in categories:
                analysis["arousal_level"] = "high"
            elif "low_arousal" in categories:
                analysis["arousal_level"] = "low"
            
            if "positive" in categories:
                analysis["valence"] = "positive"
            elif "negative" in categories:
                analysis["valence"] = "negative"
            else:
                analysis["valence"] = "neutral"
        
        return analysis


# Global instance for easy access
emotion_converter = EmotionOutputConverter()


def convert_emotion_output(model_output: Union[np.ndarray, List, Dict, Any],
                          output_format: str = "auto",
                          top_k: int = 3) -> Dict[str, Any]:
    """
    Convenience function to convert model output to standardized emotions.
    
    Args:
        model_output: Raw output from the model
        output_format: Format of the output
        top_k: Number of top emotions to return
    
    Returns:
        Dictionary with standardized emotion results
    """
    return emotion_converter.convert_model_output(model_output, output_format, top_k)


def get_emotion_summary(conversion_result: Dict[str, Any]) -> str:
    """
    Convenience function to get emotion summary.
    
    Args:
        conversion_result: Result from convert_emotion_output
    
    Returns:
        Human-readable summary string
    """
    return emotion_converter.get_emotion_summary(conversion_result)


def get_emotion_analysis(conversion_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to get emotion analysis.
    
    Args:
        conversion_result: Result from convert_emotion_output
    
    Returns:
        Dictionary with analysis information
    """
    return emotion_converter.get_emotion_analysis(conversion_result)
