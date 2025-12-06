import numpy as np

class StateTracker:
    def __init__(self, threshold=3):
        self.awake_count = 0
        self.drowsy_count = 0
        self.state = "neutral"
        self.threshold = threshold

    def update(self, label):
        if label == "drowsy":
            self.drowsy_count += 1
            self.awake_count = 0
        elif label == "awake":
            self.awake_count += 1
            self.drowsy_count = 0

        if self.drowsy_count >= self.threshold:
            self.state = "drowsy"
        elif self.awake_count >= self.threshold:
            self.state = "awake"
        else:
            self.state = "neutral"

        return self.state
