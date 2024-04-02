import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import time
import detector as dt
import shutil


# Function to save uploaded file to temp directory and return the path
def save_uploaded_file(uploaded_file):
    """Save video to temp file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
            # Write the data to the temp file
            tmpfile.write(uploaded_file.read())
            return tmpfile.name  # Return the path of the saved temp file
    except Exception as e:
        st.error(f"Failed to save file: {e}")
        return None
    
    
def process_video(video_path, detector):
    """Process video frames and save the output to a new video."""
    # Create tmp directory    
    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    # Read the video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'X264')  # libx264 (mp4v does not work with streamlit)
    out = cv2.VideoWriter('tmp/output.mp4', fourcc, fps, (width, height))
    
    # Detect defects in the video
    results = detector.detect(video_path, input_type="video")
    inf_speed = 0
    for result in results:
        # Convert PIL Image to numpy array
        processed_frame = np.array(result.plot())
        # Add frame inference time to total inference time  
        inf_speed += result.speed['inference']
        # Write the processed frame to output video
        out.write(processed_frame)
    
    # Release everything when job is finished
    cap.release()
    out.release()
    
    # Return inference speed
    return inf_speed


def remove_tmp(tmp_directory):
    try:
        shutil.rmtree(tmp_directory)
        print(f"Directory '{tmp_directory}' has been removed successfully.")
    except FileNotFoundError:
        print(f"Directory '{tmp_directory}' does not exist.")
    except Exception as e:
        print(f"Error occurred while deleting directory '{tmp_directory}': {e}")


# Get path of current directory       
dir = os.path.abspath(os.path.dirname(__file__))
probs = None

# Create detector object
dtr = dt.Detector(f"{dir}\\best.pt")



## UI 
st.title("Defect Detector")

col1, col2 = st.columns(2)

with col1:
    st.header("Input")
    types = ["jpg", "jpeg", "png","mp4","mov","wmv"]
    file = st.file_uploader("Choose an Image", type=types,key=10101)
    
    
    # Display the uploaded image
    if file is not None:
        if file.type.startswith('image'):
            in_type="image"
            input = Image.open(file)
            st.image(input, caption="Input", use_column_width=True)
        else:
            in_type="video"
            st.video(file)
            st.caption("Input")
            input = save_uploaded_file(file)
    else:
        input = None

with col2:
    st.header("Output")
    for _ in range(0,9):
        st.write("")
    
    # Make Prediction
    if file is not None:
        st.write("predicting") 
        print(f"\n{type(input)}\n")
        inf_speed = 0
        if in_type == "image":
            results = dtr.detect(input,input_type=in_type)
            summary = []
            for result in results:
                inf_speed += result.speed['inference']
                image = result.plot()
                summary.append(result.verbose()) 
                st.image(image,caption="Output", use_column_width=True)      
        elif in_type == "video":            
            inf_speed = process_video(input, dtr)
            with open("tmp/output.mp4", "rb") as file:
                st.video(data=file.read(), format="video/mp4")
            remove_tmp('tmp')
            
        st.write(f"Inference Speed: {inf_speed} ms")
    else:
        st.write("Result will show here.")
