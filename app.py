
import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
import tempfile


# Load YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load('./yolov5', 'custom', path='./yolov5_best.pt', source='local')

model = load_model()


# Object Detection Function
def detect_objects(image, model):
    results = model(image)
    detections = results.xyxy[0].cpu().numpy()  # Bounding boxes
    for det in detections:
        x_min, y_min, x_max, y_max, confidence, cls = det
        if confidence > 0.3:  # Threshold
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
            label = f"Class {int(cls)}, Conf: {confidence:.2f}"
            cv2.putText(image, label, (int(x_min), int(y_min)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Streamlit App
st.title("Object Detection with YOLOv5")
st.sidebar.title("Options")

# Sidebar options
mode = st.sidebar.radio("Select Mode", ("Image", "Video", "Real-Time"))

if mode == "Image":
    st.header("Upload an Image")
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        image_np = np.array(image)

        # Run detection
        st.write("Detecting objects...")
        detected_image = detect_objects(image_np, model)

        # Display results
        st.image(detected_image, caption="Detected Objects", use_column_width=True)

elif mode == "Video":
    st.header("Upload a Video")
    uploaded_video = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection
            frame = detect_objects(frame, model)

            # Display frame
            stframe.image(frame, channels="BGR")

        cap.release()

elif mode == "Real-Time":
    st.header("Real-Time Object Detection")
    st.write("Turn on your camera to detect objects in real-time.")

    run_detection = st.button("Start Detection")

    if run_detection:
        cap = cv2.VideoCapture(0)  # Open webcam

        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection
            frame = detect_objects(frame, model)

            # Display frame
            stframe.image(frame, channels="BGR")

            # Break the loop if 'q' is pressed (simulate this for Streamlit)
            if st.button("Stop"):
                break

        cap.release()
