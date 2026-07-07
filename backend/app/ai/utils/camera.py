import cv2

class Camera:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        # Try opening the webcam hardware link safely
        self.cap = cv2.VideoCapture(self.camera_index)
        
    def is_operational(self) -> bool:
        # Checks if the webcam opened successfully
        return self.cap.isOpened()

    def get_frame(self):
        # If the camera isn't working, return None instead of crashing
        if not self.is_operational():
            return None
            
        success, frame = self.cap.read()
        if not success:
            return None
        return frame

    def release(self):
        # Safely disconnect the camera when we are done
        if self.cap.isOpened():
            self.cap.release()