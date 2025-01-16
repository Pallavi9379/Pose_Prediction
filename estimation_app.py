import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os

# Check if the model file exists
model_path = "D:/research/projects/pose prediction/graph_opt.pb"
if not os.path.exists(model_path):
    st.error("Model file not found!")

DEMO_IMAGE = 'D:/research/projects/pose prediction/stand.jpg'

# Pose body parts and pairs
BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

# Load model
net = cv2.dnn.readNetFromTensorflow(model_path)

# Streamlit UI
st.title("Human Pose Estimation OpenCV")
st.text('Make Sure you have a clear image with all the parts clearly visible')

img_file_buffer = st.file_uploader("Upload an image, Make sure you have a clear image", type=["jpg", "jpeg", "png"])

if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
else:
    demo_image = DEMO_IMAGE
    image = np.array(Image.open(demo_image))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Display original image
st.subheader('Original Image')
st.image(image, caption="Original Image", use_column_width=True)

# Threshold slider
thres = st.slider('Threshold for detecting the key points', min_value=0.0, max_value=1.0, value=0.2, step=0.05)

# Pose detection function with caching
@st.cache_data
def poseDetector(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]
    
    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thres else None)
        
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            
    return frame

output = poseDetector(image)

# Display estimated pose
st.subheader('Positions Estimated')
st.image(output, caption="Positions Estimated", use_column_width=True)
#D:\research\projects\pose prediction\estimation_app.py