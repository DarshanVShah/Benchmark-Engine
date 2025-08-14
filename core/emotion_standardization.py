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
    
    # Core emotion categories (primary emotions)
    CORE_EMOTIONS = [
        "anger",      # 0
        "disgust",    # 1
        "fear",       # 2
        "happiness",  # 3
        "sadness",    # 4
        "surprise",   # 5
        "neutral"     # 6
    ]
    
    # Extended emotion categories (secondary emotions)
    EXTENDED_EMOTIONS = [
        "love",           # 7
        "joy",            # 8
        "trust",          # 9
        "anticipation",   # 10
        "optimism",       # 11
        "pessimism",      # 12
        "contempt",       # 13
        "embarrassment",  # 14
        "pride",          # 15
        "shame",          # 16
        "guilt",          # 17
        "amusement",      # 18
        "awe",            # 19
        "contentment",    # 20
        "excitement",     # 21
        "relief",         # 22
        "satisfaction",   # 23
        "anxiety",        # 24
        "confusion",      # 25
        "curiosity",      # 26
        "gratitude",      # 27
        "hope",           # 28
        "inspiration",    # 29
        "nostalgia",      # 30
        "wonder"          # 31
    ]
    
    # All standardized emotions
    ALL_EMOTIONS = CORE_EMOTIONS + EXTENDED_EMOTIONS
    
    # Emotion category groupings for analysis
    EMOTION_CATEGORIES = {
        "positive": ["happiness", "joy", "love", "trust", "optimism", "amusement", 
                    "awe", "contentment", "excitement", "relief", "satisfaction", 
                    "gratitude", "hope", "inspiration", "nostalgia", "wonder"],
        "negative": ["anger", "disgust", "fear", "sadness", "pessimism", "contempt", 
                    "embarrassment", "shame", "guilt", "anxiety"],
        "neutral": ["neutral", "surprise", "anticipation", "confusion", "curiosity"],
        "high_arousal": ["anger", "fear", "excitement", "anxiety", "surprise"],
        "low_arousal": ["sadness", "contentment", "relief", "satisfaction", "nostalgia"]
    }
    
    def __init__(self):
        """Initialize the standardized emotion system."""
        self.emotion_to_id = {emotion: i for i, emotion in enumerate(self.ALL_EMOTIONS)}
        self.id_to_emotion = {i: emotion for i, emotion in enumerate(self.ALL_EMOTIONS)}
        
        # Initialize mapping dictionaries for different emotion schemes
        self._initialize_mappings()
    
    def _initialize_mappings(self):
        """Initialize mappings for different emotion classification schemes."""
        
        # 2018-E-c-En dataset mapping (11 emotions)
        self.mapping_2018_ec_en = {
            "anger": "anger",
            "anticipation": "anticipation", 
            "disgust": "disgust",
            "fear": "fear",
            "joy": "joy",
            "love": "love",
            "optimism": "optimism",
            "pessimism": "pessimism",
            "sadness": "sadness",
            "surprise": "surprise",
            "trust": "trust"
        }
        
        # GoEmotions dataset mapping (27 emotions)
        # Based on the GoEmotions paper: https://arxiv.org/abs/2005.00547
        self.mapping_goemotions = {
            0: "admiration",      # Map to awe
            1: "amusement",       # Map to amusement
            2: "anger",           # Map to anger
            3: "annoyance",       # Map to anger
            4: "approval",        # Map to satisfaction
            5: "caring",          # Map to love
            6: "confusion",       # Map to confusion
            7: "curiosity",       # Map to curiosity
            8: "desire",          # Map to anticipation
            9: "disappointment",  # Map to sadness
            10: "disapproval",    # Map to contempt
            11: "disgust",        # Map to disgust
            12: "embarrassment",  # Map to embarrassment
            13: "excitement",     # Map to excitement
            14: "fear",           # Map to fear
            15: "gratitude",      # Map to gratitude
            16: "grief",          # Map to sadness
            17: "joy",            # Map to joy
            18: "love",           # Map to love
            19: "nervousness",    # Map to anxiety
            20: "optimism",       # Map to optimism
            21: "pride",          # Map to pride
            22: "realization",    # Map to surprise
            23: "relief",         # Map to relief
            24: "remorse",        # Map to guilt
            25: "sadness",        # Map to sadness
            26: "surprise",       # Map to surprise
            27: "neutral"         # Map to neutral
        }
        
        # Common emotion model mappings (e.g., HuggingFace emotion models)
        self.mapping_common_models = {
            "joy": "joy",
            "happiness": "happiness",
            "sadness": "sadness",
            "anger": "anger",
            "fear": "fear",
            "disgust": "disgust",
            "surprise": "surprise",
            "neutral": "neutral",
            "love": "love",
            "trust": "trust",
            "anticipation": "anticipation",
            "optimism": "optimism",
            "pessimism": "pessimism"
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
    
    def get_core_emotions(self) -> List[str]:
        """Get core emotions only."""
        return self.CORE_EMOTIONS.copy()
    
    def get_extended_emotions(self) -> List[str]:
        """Get extended emotions only."""
        return self.EXTENDED_EMOTIONS.copy()


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
