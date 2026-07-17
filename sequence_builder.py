import numpy as np
import collections
import time

class LiveSequenceBuilder:
    def __init__(self, sequence_length=20, num_features=63):
        """
        Manages the temporal data stage for continuous gesture models.
        
        :param sequence_length: The number of sequential frames required (e.g., N=20).
        :param num_features: Size of the normalized landmark vector (21 coordinates * 3 = 63).
        """
        self.sequence_length = sequence_length
        self.num_features = num_features
        
        # Deque naturally implements a fast, thread-safe FIFO sliding window
        self.frame_buffer = collections.deque(maxlen=self.sequence_length)

    def add_frame(self, landmark_vector):
        """
        Appends a newly processed 63D frame landmark vector to our rolling timeline history.
        """
        if landmark_vector is None or len(landmark_vector) != self.num_features:
            # Drop/ignore frames where hand tracking failed to avoid corrupting sequence motion
            return False
        
        self.frame_buffer.append(landmark_vector)
        return True

    def is_sequence_ready(self):
        """
        Returns True once the sliding window is fully populated with N frames.
        """
        return len(self.frame_buffer) == self.sequence_length

    def get_sequence_tensor(self):
        """
        Extracts the rolling frame window as a unified tensor.
        
        :return: Numpy array with shape (sequence_length, num_features) i.e. (20, 63)
        """
        if not self.is_sequence_ready():
            return None
            
        # Convert deque sequence into a contiguous 2D float tensor
        sequence_array = np.array(self.frame_buffer, dtype=np.float32)
        return sequence_array

    def clear(self):
        """Resets the history buffer when a gesture sequence finishes."""
        self.frame_buffer.clear()


# ==========================================
# 🧪 Verification Test Sandbox
# ==========================================
if __name__ == "__main__":
    print("🧪 Testing Sequence Builder Prototype...")
    
    # Initialize for a rolling window of 20 frames
    seq_builder = LiveSequenceBuilder(sequence_length=20, num_features=63)
    
    # Simulate a user performing a gesture across 25 camera frames
    for frame_idx in range(1, 26):
        # Generate a dummy normalized 63D vector (wrist at 0, slowly moving finger coordinates)
        mock_landmark_vector = np.random.rand(63).tolist()
        
        added = seq_builder.add_frame(mock_landmark_vector)
        
        if seq_builder.is_sequence_ready():
            tensor = seq_builder.get_sequence_tensor()
            print(f"Frame {frame_idx:02d}: Buffer is Full! Output Sequence Tensor shape: {tensor.shape}")
        else:
            print(f"Frame {frame_idx:02d}: Buffering frames... Current buffer size: {len(seq_builder.frame_buffer)}/{seq_builder.sequence_length}")
            
    print("\n✅ Sequence Builder Prototype verified and ready for LSTM integration!")