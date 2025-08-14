"""
TensorFlow Lite Model Adapter

This adapter handles TensorFlow Lite models for benchmarking.
Supports various input/output types and provides standardized interface.
Includes emotion standardization for consistent emotion classification.
"""

import importlib.util
import os
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np

from core.interfaces import BaseDataset

# Conditional import to avoid errors when tensorflow is not installed
if importlib.util.find_spec("tensorflow") is not None:
    import tensorflow as tf

    TENSORFLOW_AVAILABLE = True
else:
    TENSORFLOW_AVAILABLE = False
    tf = None

from core import BaseModelAdapter, DataType, ModelType, OutputType

# Import emotion standardization system
try:
    from core.emotion_standardization import standardized_emotions, map_emotion_to_standard
    from core.emotion_converter import convert_emotion_output, get_emotion_analysis
    EMOTION_STANDARDIZATION_AVAILABLE = True
except ImportError:
    EMOTION_STANDARDIZATION_AVAILABLE = False
    standardized_emotions = None
    map_emotion_to_standard = None
    convert_emotion_output = None
    get_emotion_analysis = None


class TensorFlowLiteAdapter(BaseModelAdapter):
    """
    TensorFlow Lite model adapter for benchmarking.

    """

    def __init__(self):
        self.interpreter = None
        self.model_loaded = False
        self.model_path = ""
        self.input_details = []
        self.output_details = []
        self.input_shape = None
        self.output_shape = None
        self._input_type = DataType.TEXT  # Default, can be overridden
        self._output_type = OutputType.CLASS_ID  # Default, can be overridden
        self.task_type = "auto-detected"
        self.max_length = 512
        self.tokenizer = None
        self.is_multi_label = False  # Track if this is a multi-label task
        self._load_error = None  # Track loading errors

    @property
    def input_type(self) -> DataType:
        """Return the expected input type for this model."""
        return self._input_type

    @property
    def output_type(self) -> OutputType:
        """Return the output type this model produces."""
        # Dynamic output type based on task type
        if self.is_multi_label:
            return OutputType.PROBABILITIES
        else:
            return self._output_type

    def load(self, model_path: str) -> bool:
        """Load TensorFlow Lite model from file."""
        if not TENSORFLOW_AVAILABLE:
            self._load_error = "TensorFlow is not installed. Please install tensorflow: pip install tensorflow"
            return False

        try:
            if not os.path.exists(model_path):
                self._load_error = f"Model file not found: {model_path}"
                return False

            # Load the TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()

            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            # Extract shapes
            self.input_shape = self.input_details[0]["shape"]
            self.output_shape = self.output_details[0]["shape"]

            self.model_path = model_path
            self.model_loaded = True
            
            # Clear any previous errors
            self._load_error = None

            return True

        except Exception as e:
            self._load_error = str(e)
            return False

    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure model parameters."""
        try:
            # Set task type
            if "task_type" in config:
                self.task_type = config["task_type"]

            # Set multi-label flag
            if "is_multi_label" in config:
                self.is_multi_label = config["is_multi_label"]
            elif "task_type" in config and config["task_type"] == "multi-label":
                self.is_multi_label = True

            # Set input/output types
            if "input_type" in config:
                self._input_type = DataType(config["input_type"])

            if "output_type" in config:
                self._output_type = OutputType(config["output_type"])

            # Set max length for text models
            if "max_length" in config:
                self.max_length = config["max_length"]

            return True

        except Exception as e:
            return False

    def preprocess(self, raw_input: Any) -> Any:
        """
        Convert raw input to model-specific format.

        Args:
            raw_input: Raw input from dataset

        Returns:
            Preprocessed input ready for model inference
        """
        try:
            # Determine input type and preprocess accordingly
            if self.input_type == DataType.TEXT:
                return self._preprocess_text(raw_input)
            elif self.input_type == DataType.IMAGE:
                return self._preprocess_image(raw_input)
            elif self.input_type == DataType.TENSOR:
                return self._preprocess_tensor(raw_input)
            else:
                # Default to tensor preprocessing
                return self._preprocess_tensor(raw_input)

        except Exception as e:
            return None

    def _preprocess_text(self, raw_input: Union[str, Dict[str, Any]]) -> np.ndarray:
        """Preprocess text input for TFLite model."""
        try:
            # Extract text from input
            if isinstance(raw_input, dict):
                if "text" in raw_input:
                    text = raw_input["text"]
                elif "input" in raw_input:
                    text = raw_input["input"]
                else:
                    # Try to find any string value
                    text = next(
                        (v for v in raw_input.values() if isinstance(v, str)), ""
                    )
            elif isinstance(raw_input, str):
                text = raw_input
            else:
                text = str(raw_input)

            # Handle different input tensor configurations
            if len(self.input_details) == 1:
                # Single input tensor - create a simple text representation
                input_shape = self.input_details[0]["shape"]
                if len(input_shape) == 2:  # [batch, sequence_length]
                    max_length = input_shape[1]
                    # Create a simple tokenization (character-based)
                    tokens = [ord(c) % 1000 for c in text[:max_length]]
                    # Pad or truncate to match expected shape
                    if len(tokens) < max_length:
                        tokens.extend([0] * (max_length - len(tokens)))
                    elif len(tokens) > max_length:
                        tokens = tokens[:max_length]
                    
                    return np.array([tokens], dtype=np.int32)
                else:
                    # Handle other shapes by flattening
                    return np.zeros(input_shape, dtype=np.int32)
            
            elif len(self.input_details) == 3:
                # Multiple input tensors (attention_mask, input_ids, token_type_ids)
                # attention_mask: Use actual shape from model
                attention_mask_shape = self.input_details[0]["shape"]
                attention_mask = np.ones(attention_mask_shape, dtype=np.int32)
                
                # input_ids: Use actual shape from model
                input_ids_shape = self.input_details[1]["shape"]
                input_ids = np.zeros(input_ids_shape, dtype=np.int32)
                
                # For very short sequences (like [1, 1]), we need to handle differently
                if len(input_ids_shape) == 2 and input_ids_shape[1] == 1:
                    # Model expects single token - use first character as token ID
                    if text:
                        input_ids[0, 0] = ord(text[0]) % 1000  # Simple hash
                    else:
                        input_ids[0, 0] = 0
                else:
                    # Model expects longer sequences - truncate/pad to match
                    max_length = input_ids_shape[1] if len(input_ids_shape) > 1 else 1
                    if len(text) > max_length:
                        text = text[:max_length]
                    
                    for i, char in enumerate(text):
                        if i < max_length:
                            input_ids[0, i] = ord(char) % 1000
                
                # token_type_ids: Use actual shape from model
                token_type_ids_shape = self.input_details[2]["shape"]
                token_type_ids = np.zeros(token_type_ids_shape, dtype=np.int32)
                
                # Return as a list of tensors matching the model's input format
                return [attention_mask, input_ids, token_type_ids]
            
            else:
                # Fallback: create dummy input matching the first input tensor
                input_shape = self.input_details[0]["shape"]
                return np.zeros(input_shape, dtype=np.int32)

        except Exception as e:
            # Return dummy tensors with correct shapes from model
            if len(self.input_details) == 3:
                return [
                    np.ones(self.input_details[0]["shape"], dtype=np.int32),  # attention_mask
                    np.zeros(self.input_details[1]["shape"], dtype=np.int32),  # input_ids
                    np.zeros(self.input_details[2]["shape"], dtype=np.int32)   # token_type_ids
                ]
            else:
                return np.zeros(self.input_details[0]["shape"], dtype=np.int32)

    def _preprocess_image(self, raw_input: Union[str, Dict[str, Any]]) -> np.ndarray:
        """Preprocess image input for TFLite model."""
        try:
            # Extract image path or data
            if isinstance(raw_input, dict):
                if "image" in raw_input:
                    image_path = raw_input["image"]
                elif "path" in raw_input:
                    image_path = raw_input["path"]
                else:
                    raise ValueError("No image path found in input")
            elif isinstance(raw_input, str):
                image_path = raw_input
            else:
                raise ValueError("Invalid image input format")

            # Load and preprocess image
            # This is a simplified implementation
            # In a real implementation, you'd use proper image loading and preprocessing
            try:
                import cv2

                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not load image: {image_path}")

                # Resize to expected input shape
                if len(self.input_shape) == 4:  # [batch, height, width, channels]
                    target_height, target_width = (
                        self.input_shape[1],
                        self.input_shape[2],
                    )
                else:
                    target_height, target_width = 224, 224  # Default size

                resized = cv2.resize(image, (target_width, target_height))

                # Normalize pixel values
                normalized = resized.astype(np.float32) / 255.0

                # Add batch dimension if needed
                if len(normalized.shape) == 3:
                    normalized = np.expand_dims(normalized, axis=0)

                return normalized
            except ImportError:
                # OpenCV not available, using dummy image data
                return np.random.rand(*self.input_shape).astype(np.float32)

        except Exception as e:
            return np.zeros(self.input_shape, dtype=np.float32)

    def _preprocess_tensor(self, raw_input: Any) -> np.ndarray:
        """Preprocess tensor input for TFLite model."""
        try:
            # Convert to numpy array
            if isinstance(raw_input, np.ndarray):
                tensor = raw_input
            elif isinstance(raw_input, (list, tuple)):
                tensor = np.array(raw_input)
            else:
                tensor = np.array([raw_input])

            # Ensure correct shape
            if tensor.shape != self.input_shape:
                # Try to reshape
                try:
                    tensor = tensor.reshape(self.input_shape)
                except ValueError:
                    # Pad or truncate to match expected shape
                    if len(tensor.shape) == 1 and len(self.input_shape) == 2:
                        # Add batch dimension
                        tensor = np.expand_dims(tensor, axis=0)

            return tensor.astype(np.float32)

        except Exception as e:
            return np.zeros(self.input_shape, dtype=np.float32)

    def run(
        self, preprocessed_input: Union[np.ndarray, List[np.ndarray]]
    ) -> np.ndarray:
        """
        Run inference on preprocessed input.

        Args:
            preprocessed_input: Preprocessed input from preprocess()
            (can be single tensor or list of tensors)

        Returns:
            Raw model output
        """
        try:
            if not self.model_loaded:
                return None

            # Handle multiple input tensors
            if isinstance(preprocessed_input, list):
                # Set multiple input tensors
                for i, tensor in enumerate(preprocessed_input):
                    if i < len(self.input_details):
                        self.interpreter.set_tensor(
                            self.input_details[i]["index"], tensor
                        )
                    else:
                        break
            else:
                # Single input tensor
                self.interpreter.set_tensor(
                    self.input_details[0]["index"], preprocessed_input
                )

            # Run inference
            self.interpreter.invoke()

            # Get output
            output_tensor = self.interpreter.get_tensor(self.output_details[0]["index"])
            return output_tensor

        except Exception as e:
            return None

    def postprocess(self, model_output: np.ndarray) -> Any:
        """
        Convert raw model output to standardized format.

        Args:
            model_output: Raw output from model

        Returns:
            Standardized prediction
        """
        try:
            if model_output is None:
                return None

            # Convert to standardized format based on output type
            if self.output_type == OutputType.CLASS_ID:
                # For classification, return class IDs
                predicted_class = np.argmax(model_output, axis=-1)
                return predicted_class.tolist()

            elif self.output_type == OutputType.PROBABILITIES:
                # For probabilities, return as-is
                return model_output.tolist()

            elif self.output_type == OutputType.LOGITS:
                # For logits, return as-is
                return model_output.tolist()

            else:
                # Default: return as-is
                return model_output.tolist()

        except Exception as e:
            return None

    def convert_to_standardized_emotions(self, model_output: np.ndarray, 
                                       output_format: str = "auto") -> Dict[str, Any]:
        """
        Convert model output to standardized emotions with analysis.
        
        Args:
            model_output: Raw model output
            output_format: Output format ("probabilities", "logits", "multi_label", or "auto")
            
        Returns:
            Dictionary with standardized emotion results and analysis
        """
        if not EMOTION_STANDARDIZATION_AVAILABLE:
            return {
                "error": "Emotion standardization not available",
                "raw_output": model_output.tolist() if model_output is not None else None
            }
        
        try:
            if model_output is None:
                return {"error": "No model output to convert"}
            
            # Auto-detect output format if not specified
            if output_format == "auto":
                if self.output_type == OutputType.PROBABILITIES:
                    output_format = "probabilities"
                elif self.output_type == OutputType.LOGITS:
                    output_format = "logits"
                else:
                    output_format = "multi_label"
            
            # Convert to standardized emotions
            result = convert_emotion_output(model_output, output_format, top_k=3)
            
            # Add emotion analysis
            if result and "top_emotion" in result:
                analysis = get_emotion_analysis(result)
                result["analysis"] = analysis
                
                # Add emotion standardization info
                result["emotion_standardization"] = {
                    "available_emotions": len(standardized_emotions.get_all_emotions()),
                    "emotion_set": standardized_emotions.get_all_emotions(),
                    "conversion_format": output_format
                }
            
            return result
            
        except Exception as e:
            return {
                "error": f"Error converting to standardized emotions: {str(e)}",
                "raw_output": model_output.tolist() if model_output is not None else None
            }

    def get_emotion_mapping_info(self) -> Dict[str, Any]:
        """
        Get information about emotion mapping capabilities.
        
        Returns:
            Dictionary with emotion mapping information
        """
        if not EMOTION_STANDARDIZATION_AVAILABLE:
            return {
                "emotion_standardization_available": False,
                "error": "Emotion standardization not available"
            }
        
        try:
            return {
                "emotion_standardization_available": True,
                "standardized_emotions": standardized_emotions.get_all_emotions(),
                "emotion_categories": standardized_emotions.EMOTION_CATEGORIES,
                "supported_schemes": ["2018_ec_en", "goemotions", "common_models", "auto"],
                "total_emotions": len(standardized_emotions.get_all_emotions())
            }
        except Exception as e:
            return {
                "emotion_standardization_available": False,
                "error": f"Error getting emotion mapping info: {str(e)}"
            }

    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded model."""
        info = {
            "name": os.path.basename(self.model_path),
            "type": "tensorflow_lite",
            "path": self.model_path,
            "loaded": self.model_loaded,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "input_type": self.input_type.value,
            "output_type": self.output_type.value,
            "task_type": self.task_type,
            "load_error": self._load_error,
        }

        # Add emotion standardization information if available
        if EMOTION_STANDARDIZATION_AVAILABLE:
            emotion_info = self.get_emotion_mapping_info()
            info["emotion_standardization"] = emotion_info

        return info

    def get_model_type(self) -> ModelType:
        """Return the type of model."""
        return ModelType.TENSORFLOW_LITE

    def validate_compatibility(self, dataset: "BaseDataset") -> bool:
        """Validate that this model is compatible with the given dataset."""
        return dataset.output_type == self.input_type

    def get_load_error(self) -> Optional[str]:
        """Return the last loading error if any."""
        return self._load_error
