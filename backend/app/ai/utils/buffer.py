from collections import deque

class BufferManager:
    def __init__(self, max_frames=10, threshold=7):
        """
        Manages rolling frame history to eliminate gesture flickering.
        - max_frames: Number of past frames to remember.
        - threshold: Minimum identical frames needed to update the final sign.
        """
        self.history = deque(maxlen=max_frames)
        self.threshold = threshold
        self.current_stable_sign = "Scanning..."

    def update(self, raw_sign):
        """Adds a raw prediction and returns the stabilized result."""
        if raw_sign == "No Hand Detected":
            self.history.clear()
            self.current_stable_sign = "No Hand Detected"
            return self.current_stable_sign

        # Add current frame result to rolling window
        self.history.append(raw_sign)

        # Wait until the history buffer has enough data points
        if len(self.history) == self.history.maxlen:
            # Find the most frequently occurring gesture in our window
            most_frequent = max(set(self.history), key=self.history.count)
            
            # Change display only if it beats our threshold confidence
            if self.history.count(most_frequent) >= self.threshold:
                self.current_stable_sign = most_frequent
                
        return self.current_stable_sign