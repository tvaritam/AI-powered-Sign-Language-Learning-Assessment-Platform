import os
import cv2
import time
import json
from datetime import datetime
from backend.app.ai.utils.camera import Camera
from backend.app.ai.hand_tracking.tracker import HandTracker
from backend.app.ai.gesture_recognition.detector import SignDetector
from backend.app.ai.utils.buffer import BufferManager

def run_final_sprint_interface():
    cam = Camera(0)
    tracker = HandTracker()
    detector = SignDetector()
    buffer_mgr = BufferManager(max_frames=10, threshold=7)
    
    # Create the captures directory if it doesn't exist yet
    os.makedirs("captures", exist_ok=True)
    capture_count = 0

    if not cam.is_operational():
        print("[ERROR] Camera is not operational.")
        return

    print("[SUCCESS] Pipeline running. Press 'S' to save data snapshot. Press 'Q' to quit.")
    prev_time = time.time()

    while True:
        frame = cam.get_frame()
        if frame is None:
            break

        frame = cv2.flip(frame, 1)
        ai_results = tracker.process_frame(frame)
        frame, hand_count, all_hands_data = tracker.draw_landmarks(frame, ai_results)

        # Track stable sign predictions
        if hand_count > 0:
            raw_sign = detector.recognize_static_letter(all_hands_data[0])
            stable_sign = buffer_mgr.update(raw_sign)
        else:
            stable_sign = buffer_mgr.update("No Hand Detected")

        # Performance Calculations
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time))
        prev_time = curr_time

        # Render Text Elements
        cv2.putText(frame, f"FPS: {fps}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if stable_sign == "No Hand Detected":
            cv2.putText(frame, stable_sign, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"Stable Sign: {stable_sign}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Weekend Sprint Interface", frame)
        
        # Check Keypress Commands
        key = cv2.waitKey(1) & 0xFF
        
        # If user presses 'S' and a hand is visible, export to JSON file
        if key == ord('s'):
            if hand_count > 0:
                capture_count += 1
                filename = f"captures/capture_{capture_count:03d}.json"
                
                # Format exactly like assignment sheet requested
                snapshot_data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "hand_number": hand_count,
                    "landmarks": all_hands_data[0] # Saves all 21 (x, y, z) lists
                }
                
                with open(filename, "w") as json_file:
                    json.dump(snapshot_data, json_file, indent=4)
                print(f"[SAVED] Coordinates exported successfully to {filename}")
            else:
                print("[WARNING] No hand detected on screen to capture data from.")

        elif key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_final_sprint_interface()