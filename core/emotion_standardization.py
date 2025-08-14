"""
Standardized Emotion Mapping System for BenchmarkEngine

This module provides a standardized set of emotions and mapping functions to ensure
consistent emotion labels across different models and datasets. It handles the mapping
between various emotion classification schemes and our standardized emotion set.
"""

from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class StandardizedEmotions:
    """
    Standardized emotion classification system for consistent benchmarking.
    
    This class defines a core set of emotions that can be mapped to various
    emotion classification schemes used by different models and datasets.
    """
    
    # Core emotion categories (primary emotions) - 8 emotions total
    ALL_EMOTIONS = [
        "anger",      # 0 - Strong negative emotion, often with aggression
        "disgust",    # 1 - Aversion or repulsion
        "fear",       # 2 - Anxiety or apprehension
        "happiness",  # 3 - Positive emotional state, contentment
        "sadness",    # 4 - Negative emotional state, sorrow
        "surprise",   # 5 - Sudden reaction to unexpected events
        "love",       # 6 - Affection and attachment
        "neutral"     # 7 - Absence of strong emotion
    ]
    
    # Emotion category groupings for analysis
    EMOTION_CATEGORIES = {
        "positive": ["happiness", "love"],
        "negative": ["anger", "disgust", "fear", "sadness"],
        "neutral": ["neutral", "surprise"],
        "high_arousal": ["anger", "fear", "surprise"],
        "low_arousal": ["sadness", "happiness", "love"]
    }
    
    def __init__(self):
        """Initialize the standardized emotion system."""
        self.emotion_to_id = {emotion: i for i, emotion in enumerate(self.ALL_EMOTIONS)}
        self.id_to_emotion = {i: emotion for i, emotion in enumerate(self.ALL_EMOTIONS)}
        
        # Initialize mapping dictionaries for different emotion schemes
        self._initialize_mappings()
    
    def _initialize_mappings(self):
        """Initialize mappings for different emotion classification schemes."""
        
        # 2018-E-c-En dataset mapping (11 emotions) - mapped to our 8 standardized emotions
        self.mapping_2018_ec_en = {
            "anger": "anger",           # anger -> anger
            "anticipation": "neutral",  # anticipation -> neutral (uncertainty)
            "disgust": "disgust",       # disgust -> disgust
            "fear": "fear",             # fear -> fear
            "joy": "happiness",         # joy -> happiness (positive feeling)
            "love": "love",             # love -> love
            "optimism": "happiness",    # optimism -> happiness (positive feeling)
            "pessimism": "sadness",     # pessimism -> sadness (negative feeling)
            "sadness": "sadness",       # sadness -> sadness
            "surprise": "surprise",     # surprise -> surprise
            "trust": "happiness"        # trust -> happiness (positive feeling)
        }
        
        # GoEmotions dataset mapping (28 emotions) - mapped to our 8 standardized emotions
        # Based on the GoEmotions paper: https://arxiv.org/abs/2005.00547
        self.mapping_goemotions = {
            0: "happiness",       # admiration -> happiness (positive feeling)
            1: "happiness",       # amusement -> happiness (positive feeling)
            2: "anger",           # anger -> anger
            3: "anger",           # annoyance -> anger (similar negative emotion)
            4: "happiness",       # approval -> happiness (positive feeling)
            5: "love",            # caring -> love (affection)
            6: "neutral",         # confusion -> neutral (uncertainty)
            7: "neutral",         # curiosity -> neutral (interest)
            8: "love",            # desire -> love (affection)
            9: "sadness",         # disappointment -> sadness
            10: "anger",          # disapproval -> anger (negative reaction)
            11: "disgust",        # disgust -> disgust
            12: "fear",           # embarrassment -> fear (anxiety)
            13: "happiness",      # excitement -> happiness (positive feeling)
            14: "fear",           # fear -> fear
            15: "happiness",      # gratitude -> happiness (positive feeling)
            16: "sadness",        # grief -> sadness
            17: "happiness",      # joy -> happiness (positive feeling)
            18: "love",           # love -> love
            19: "fear",           # nervousness -> fear (anxiety)
            20: "happiness",      # optimism -> happiness (positive feeling)
            21: "happiness",      # pride -> happiness (positive feeling)
            22: "surprise",       # realization -> surprise
            23: "happiness",      # relief -> happiness (positive feeling)
            24: "sadness",        # remorse -> sadness
            25: "sadness",        # sadness -> sadness
            26: "surprise",       # surprise -> surprise
            27: "neutral"         # neutral -> neutral
        }
        
        # Common emotion model mappings (e.g., HuggingFace emotion models)
        # Mapped to our 8 standardized emotions
        self.mapping_common_models = {
            "joy": "happiness",        # joy -> happiness (positive feeling)
            "happiness": "happiness",  # happiness -> happiness
            "sadness": "sadness",      # sadness -> sadness
            "anger": "anger",          # anger -> anger
            "fear": "fear",            # fear -> fear
            "disgust": "disgust",      # disgust -> disgust
            "surprise": "surprise",    # surprise -> surprise
            "neutral": "neutral",      # neutral -> neutral
            "love": "love",            # love -> love
            "trust": "happiness",      # trust -> happiness (positive feeling)
            "anticipation": "neutral", # anticipation -> neutral (uncertainty)
            "optimism": "happiness",   # optimism -> happiness (positive feeling)
            "pessimism": "sadness"     # pessimism -> sadness (negative feeling)
        }
    
    def get_standardized_emotion(self, emotion_input: Union[str, int], 
                                source_scheme: str = "auto") -> Tuple[str, float]:
        """
        Convert an emotion from any scheme to our standardized emotion.
        
        Args:
            emotion_input: The emotion to convert (string or numeric ID)
            source_scheme: The source emotion scheme ("2018_ec_en", "goemotions", 
                          "common_models", or "auto" for automatic detection)
        
        Returns:
            Tuple of (standardized_emotion, confidence_score)
        """
        if source_scheme == "auto":
            source_scheme = self._detect_scheme(emotion_input)
        
        try:
            if source_scheme == "2018_ec_en":
                return self._map_2018_ec_en(emotion_input)
            elif source_scheme == "goemotions":
                return self._map_goemotions(emotion_input)
            elif source_scheme == "common_models":
                return self._map_common_models(emotion_input)
            else:
                # Try to find the closest match
                return self._find_closest_emotion(emotion_input)
        except Exception as e:
            logger.warning(f"Error mapping emotion {emotion_input}: {e}")
            return "neutral", 0.0
    
    def _detect_scheme(self, emotion_input: Union[str, int]) -> str:
        """Automatically detect the emotion scheme based on input."""
        if isinstance(emotion_input, int):
            if 0 <= emotion_input <= 27:
                return "goemotions"
            else:
                return "common_models"
        elif isinstance(emotion_input, str):
            emotion_lower = emotion_input.lower()
            if emotion_lower in self.mapping_2018_ec_en:
                return "2018_ec_en"
            elif emotion_lower in self.mapping_common_models:
                return "common_models"
            else:
                return "common_models"
        return "common_models"
    
    def _map_2018_ec_en(self, emotion: str) -> Tuple[str, float]:
        """Map 2018-E-c-En emotions to standardized emotions."""
        emotion_lower = emotion.lower()
        if emotion_lower in self.mapping_2018_ec_en:
            standardized = self.mapping_2018_ec_en[emotion_lower]
            return standardized, 1.0
        return "neutral", 0.0
    
    def _map_goemotions(self, emotion_id: int) -> Tuple[str, float]:
        """Map GoEmotions numeric IDs to standardized emotions."""
        if emotion_id in self.mapping_goemotions:
            standardized = self.mapping_goemotions[emotion_id]
            return standardized, 1.0
        return "neutral", 0.0
    
    def _map_common_models(self, emotion: str) -> Tuple[str, float]:
        """Map common emotion model outputs to standardized emotions."""
        emotion_lower = emotion.lower()
        if emotion_lower in self.mapping_common_models:
            standardized = self.mapping_common_models[emotion_lower]
            return standardized, 1.0
        return "neutral", 0.0
    
    def _find_closest_emotion(self, emotion_input: Union[str, int]) -> Tuple[str, float]:
        """
        Find the closest standardized emotion using fuzzy matching.
        
        Args:
            emotion_input: The emotion to find a match for
        
        Returns:
            Tuple of (closest_emotion, confidence_score)
        """
        if isinstance(emotion_input, str):
            emotion_lower = emotion_input.lower()
            
            # Exact match
            if emotion_lower in self.ALL_EMOTIONS:
                return emotion_lower, 1.0
            
            # Partial match
            for std_emotion in self.ALL_EMOTIONS:
                if emotion_lower in std_emotion or std_emotion in emotion_lower:
                    return std_emotion, 0.8
            
            # Similarity matching (simple string similarity)
            best_match = "neutral"
            best_score = 0.0
            
            for std_emotion in self.ALL_EMOTIONS:
                score = self._calculate_similarity(emotion_lower, std_emotion)
                if score > best_score:
                    best_score = score
                    best_match = std_emotion
            
            if best_score > 0.3:  # Threshold for similarity
                return best_match, best_score
            else:
                return "neutral", 0.0
        
        return "neutral", 0.0
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate simple string similarity between two emotions."""
        # Simple character-based similarity
        common_chars = set(str1) & set(str2)
        total_chars = set(str1) | set(str2)
        
        if not total_chars:
            return 0.0
        
        return len(common_chars) / len(total_chars)
    
    def get_emotion_categories(self, emotion: str) -> List[str]:
        """Get the categories that an emotion belongs to."""
        categories = []
        for category, emotions in self.EMOTION_CATEGORIES.items():
            if emotion in emotions:
                categories.append(category)
        return categories
    
    def get_emotions_in_category(self, category: str) -> List[str]:
        """Get all emotions in a specific category."""
        return self.EMOTION_CATEGORIES.get(category, [])
    
    def get_emotion_id(self, emotion: str) -> Optional[int]:
        """Get the numeric ID for a standardized emotion."""
        return self.emotion_to_id.get(emotion)
    
    def get_emotion_name(self, emotion_id: int) -> Optional[str]:
        """Get the emotion name for a numeric ID."""
        return self.id_to_emotion.get(emotion_id)
    
    def get_all_emotions(self) -> List[str]:
        """Get all standardized emotions."""
        return self.ALL_EMOTIONS.copy()
    



# Global instance for easy access
standardized_emotions = StandardizedEmotions()


def map_emotion_to_standard(emotion_input: Union[str, int], 
                           source_scheme: str = "auto") -> Tuple[str, float]:
    """
    Convenience function to map any emotion to standardized format.
    
    Args:
        emotion_input: The emotion to convert
        source_scheme: The source emotion scheme
    
    Returns:
        Tuple of (standardized_emotion, confidence_score)
    """
    return standardized_emotions.get_standardized_emotion(emotion_input, source_scheme)


def get_standardized_emotion_list() -> List[str]:
    """Get the complete list of standardized emotions."""
    return standardized_emotions.get_all_emotions()


def get_emotion_categories(emotion: str) -> List[str]:
    """Get the categories that an emotion belongs to."""
    return standardized_emotions.get_emotion_categories(emotion)
