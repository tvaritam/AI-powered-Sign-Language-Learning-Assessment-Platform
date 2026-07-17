import os
import time
import logging
import pickle
import numpy as np
import cv2
import urllib.request
import ctypes

# ===================================================
# GLOBAL WINDOWS MEDIAPIPE PYTHON 3.13 BUG PATCH
# ===================================================
# Intercepts ctypes at the system level to bypass the Windows 
# "AttributeError: function 'free' not found" issue in MediaPipe Tasks.
original_getattr = ctypes.CDLL.__getattr__
original_getitem = ctypes.CDLL.__getitem__

class MockFree:
    def __init__(self):
        self.argtypes = []
    def __call__(self, *args, **kwargs):
        pass

def patched_getattr(self, name):
    if name == 'free':
        return MockFree()
    return original_getattr(self, name)

def patched_getitem(self, name):
    if name == 'free':
        return MockFree()
    return original_getitem(self, name)

ctypes.CDLL.__getattr__ = patched_getattr
ctypes.CDLL.__getitem__ = patched_getitem
# ===================================================

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Configure structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AI_Inference_Engine")

class InferenceEngine:
    def __init__(self, model_version="rf_v1", confidence_threshold=0.85):
        """
        Initializes the self-contained AI Pipeline.
        Handles model loading and MediaPipe Tasks.
        """
        self.model_version = model_version
        self.confidence_threshold = confidence_threshold
        
        # 1. Load the frozen Random Forest classifier
        self.model_dir = "backend/app/ai"
        self.model_path = os.path.join(self.model_dir, "randomforest.pkl")
        self.model = self._load_model()

        # 2. Initialize MediaPipe HandLandmarker Tasks API
        self.mp_model_path = "hand_landmarker.task"
        self._ensure_mp_model_exists()
        
        base_options = python.BaseOptions(model_asset_path=self.mp_model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,  # Set to 1 to reject multi-hand interference as requested
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def _ensure_mp_model_exists(self):
        """Downloads the MediaPipe task model if not present locally."""
        if not os.path.exists(self.mp_model_path):
            logger.info(f"Downloading MediaPipe task asset to '{self.mp_model_path}'...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, self.mp_model_path)
            logger.info("MediaPipe task asset downloaded successfully.")

    def _load_model(self):
        """Safely loads the serialized machine learning model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ ML Model not found at {self.model_path}!")
        with open(self.model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"✅ Loaded ML model version [{self.model_version}] successfully.")
        return model

    def _validate_and_normalize(self, hand_landmarks):
        """
        Landmark Validation & Relative Translation Normalization.
        Transforms coordinates relative to the wrist (landmark 0).
        """
        # Ensure we have exactly 21 landmarks for a complete hand shape
        if len(hand_landmarks) != 21:
            return None

        coords = []
        for lm in hand_landmarks:
            coords.extend([lm.x, lm.y, lm.z])
            
        coords = np.array(coords).reshape(-1, 3)
        wrist = coords[0] # The wrist (landmark 0) is our anchor
        
        # Translate landmarks so wrist becomes (0,0,0)
        normalized_coords = coords - wrist
        
        return normalized_coords.flatten() # Generates a 63-Dimensional Feature Vector

    def predict(self, frame_bgr):
        """
        Unified public API. Accepts raw OpenCV frame, handles full pipeline,
        and returns a standardized prediction dictionary contract.
        """
        start_time = time.time()
        
        if frame_bgr is None:
            return self._build_response("Invalid Input Frame", 0.0, "REJECTED_INVALID_FRAME", 0.0)

        # 1. Image preprocessing
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # 2. Run MediaPipe Landmarker detection
        detection_result = self.detector.detect(mp_image)
        
        # Frame validation: Reject frames with no hands detected
        if not detection_result.hand_landmarks:
            inference_time = (time.time() - start_time) * 1000
            return self._build_response(
                label="No Hand Detected", 
                confidence=0.0, 
                status="REJECTED_NO_HAND", 
                inference_time_ms=round(inference_time, 2)
            )
            
        # 3. Extract, Validate, and Normalize the 63D landmark vector
        first_hand = detection_result.hand_landmarks[0]
        feature_vector = self._validate_and_normalize(first_hand)
        
        if feature_vector is None:
            inference_time = (time.time() - start_time) * 1000
            return self._build_response(
                label="Incomplete Landmarks", 
                confidence=0.0, 
                status="REJECTED_INCOMPLETE_LANDMARKS", 
                inference_time_ms=round(inference_time, 2)
            )
            
        # 4. Model Inference & Probability scoring
        feature_vector_reshaped = feature_vector.reshape(1, -1)
        prediction_class = self.model.predict(feature_vector_reshaped)[0]
        probabilities = self.model.predict_proba(feature_vector_reshaped)[0]
        confidence = float(np.max(probabilities))
        
        # 5. Apply confidence threshold rules
        status = "ACCEPTED"
        final_label = prediction_class
        if confidence < self.confidence_threshold:
            status = "REJECTED_LOW_CONFIDENCE"
            final_label = f"Unclear (Best Guess: {prediction_class})"
            
        inference_time = (time.time() - start_time) * 1000
        
        return self._build_response(
            label=final_label, 
            confidence=round(confidence, 4), 
            status=status, 
            inference_time_ms=round(inference_time, 2)
        )

    def _build_response(self, label, confidence, status, inference_time_ms):
        """Constructs a strict, standardized python output dictionary contract."""
        return {
            "prediction": label,
            "confidence": confidence,
            "status": status,
            "inference_time_ms": inference_time_ms,
            "model_version": self.model_version,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }