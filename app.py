#run with streamlit run app.py
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import av
import cv2
from ultralytics import YOLO
from utils import StateTracker
import numpy as np

# Load model
model = YOLO("runs/detect/train/weights/last.pt")  # Replace with your trained model path

# Streamlit app UI
st.set_page_config(layout="centered")
st.title("🚗Drowsiness Detection System")
st.markdown("Detects if you're drowsy while driving.")

alert_placeholder = st.empty()
st.session_state.alert = False  # Set default alert state

# Tracker class to monitor consecutive predictions
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.state_tracker = StateTracker(threshold=5)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model(img, verbose=False)[0]
        labels = results.names
        boxes = results.boxes
        detections = results.boxes.cls.cpu().numpy() if boxes else []

        pred_label = None
        if len(detections) > 0:
            pred_index = int(detections[0])
            pred_label = labels[pred_index]
        else:
            pred_label = "none"

        state = self.state_tracker.update(pred_label)

        if state == "drowsy":
            cv2.putText(img, "DROWSY ALERT!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            st.session_state.alert = True
        elif state == "awake":
            cv2.putText(img, "AWAKE", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            st.session_state.alert = False
        else:
            cv2.putText(img, "Detecting...", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
            st.session_state.alert = False
        return img

# Start webcam stream
webrtc_streamer(key="drowsiness", video_processor_factory=VideoTransformer)

# Visual alert message
if st.session_state.alert:
    alert_placeholder.error("🚨 Drowsiness Detected! Please stay alert!", icon="⚠️")
else:
    alert_placeholder.info("Monitoring...", icon="✅")
