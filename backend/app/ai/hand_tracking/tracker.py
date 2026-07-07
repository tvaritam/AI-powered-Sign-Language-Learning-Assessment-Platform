import os
import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.7):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "hand_landmarker.task")
        
        if not os.path.exists(model_path):
            print(f"[ERROR] Missing model file at: {model_path}")

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence
        )
        self.detector = HandLandmarker.create_from_options(options)

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.detector.detect(mp_image)

    def draw_landmarks(self, frame, results):
        hand_count = 0
        all_hands_data = [] # Store coordinates for all detected hands
        
        if results and results.hand_landmarks:
            hand_count = len(results.hand_landmarks)
            height, width, _ = frame.shape
            
            for hand_landmarks in results.hand_landmarks:
                hand_coords = [] # Store 21 joints for THIS hand
                
                for landmark in hand_landmarks:
                    # 1. Convert normalized 0.0-1.0 coordinates to actual pixel positions
                    cx, cy = int(landmark.x * width), int(landmark.y * height)
                    cz = landmark.z # Depth relative to the wrist
                    
                    # 2. Save to our data list
                    hand_coords.append((cx, cy, cz))
                    
                    # 3. Draw the joint dot
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                
                all_hands_data.append(hand_coords)
                    
        return frame, hand_count, all_hands_data