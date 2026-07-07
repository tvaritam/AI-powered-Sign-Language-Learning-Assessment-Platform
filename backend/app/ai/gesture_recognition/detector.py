class SignDetector:
    def __init__(self):
        # MediaPipe landmark IDs for the 4 finger tips and their corresponding base joints
        self.finger_tips = [8, 12, 16, 20]      # Index, Middle, Ring, Pinky tips
        self.finger_pip_joints = [6, 10, 14, 18] # Middle joint of each finger

    def recognize_static_letter(self, hand_coords):
        """
        Takes a list of 21 coordinates [(x, y, z), ...] for one hand
        and returns a string indicating the detected letter or gesture.
        """
        if not hand_coords or len(hand_coords) < 21:
            return "No Data"

        # Track which fingers are pointing UP (True = Up, False = Down)
        # Note: In OpenCV, Y decreases as you move UP the screen.
        opened_fingers = []
        
        # Check the 4 standard fingers
        for tip, pip in zip(self.finger_tips, self.finger_pip_joints):
            if hand_coords[tip][1] < hand_coords[pip][1]:  # Tip is higher than middle joint
                opened_fingers.append(True)
            else:
                opened_fingers.append(False)

        # Check the Thumb (uses X-axis movement instead of Y)
        # If thumb tip is further out horizontally than its base joint, it's open
        thumb_tip = hand_coords[4][0]
        thumb_ip = hand_coords[3][0]
        thumb_is_open = thumb_tip > thumb_ip  # Assuming right hand facing camera
        
        # --- GESTURE LOGIC RULES ---
        
        # 1. Open Hand (All fingers up) -> "HELLO" or "5"
        if all(opened_fingers) and thumb_is_open:
            return "HELLO"
            
        # 2. Only Index and Middle up -> "VICTORY" or "V"
        elif opened_fingers == [True, True, False, False]:
            return "VICTORY"
            
        # 3. Only Pinky and Thumb up -> "SPIDERMAN" or "ILY"
        elif opened_fingers == [False, False, False, True] and thumb_is_open:
            return "I LOVE YOU"
            
        # 4. All fingers curled down -> "FIST" or "A"
        elif not any(opened_fingers) and not thumb_is_open:
            return "FIST"

        return "Scanning..."