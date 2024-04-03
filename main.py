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


def clear():
    if 'in_file' in st.session_state:
        del st.session_state['in_file']
    if 'out_file' in st.session_state:
        del st.session_state['out_file']
    if 'inf_speed' in st.session_state:
        st.session_state['inf_speed'] = 0
    if 'summary' in st.session_state:
        st.session_state['summary'] = []



# Get path of current directory       
dir = os.path.abspath(os.path.dirname(__file__))
probs = None

# Create detector object
dtr = dt.Detector(f"{dir}\\best.pt")



## UI 
st.title("Defect Detector")

col1, col2 = st.columns(2)

if 'in_file' not in st.session_state:
    st.session_state['in_file'] = None
if 'out_file' not in st.session_state:
    st.session_state['out_file'] = None
if 'inf_speed' not in st.session_state:
    st.session_state['inf_speed'] = 0
if 'summary' not in st.session_state:
    st.session_state['summary'] = []

with col1:
    st.header("Input")
    types = ["jpg", "jpeg", "png","mp4","mov","wmv"]
    
    
    file = st.session_state.in_file
    if file is None:
        file = st.file_uploader("Choose an Image", type=types, key='file_uploader')
        st.session_state.in_file = file
        if file is not None:
            st.rerun()
    
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
    
    # Make Prediction
    if file is not None: 
        print(f"\n{type(input)}\n")
        if in_type == "image":
            if st.session_state.out_file is None:
                output = dtr.detect(input,input_type=in_type)
                st.session_state.out_file = output
            
            results = st.session_state.out_file
            st.session_state.inf_speed = 0
            for result in results:
                st.session_state.inf_speed += result.speed['inference']
                image = result.plot()
                st.session_state.summary.append(result.verbose()) 
                st.image(image,caption="Output", use_column_width=True)      
        elif in_type == "video":       
            if st.session_state.out_file is None:
                st.spinner(text="Detection in progress...")    
                st.session_state.inf_speed = process_video(input, dtr)
                with open("tmp/output.mp4", "rb") as file:
                    st.session_state.out_file = file.read()
            st.video(st.session_state.out_file, format="video/mp4")
            
            remove_tmp('tmp')
    else:
        st.write("Result will show here.")
        st.session_state.inf_speed = 0
        st.session_state.summary = []

dashboard = st.container(border=True)

css = open('style.css')
style = f"""
<style>
    {css.read()}
</style>
"""
dashboard.markdown(style, unsafe_allow_html=True)
dashboard.markdown("<div class='dashboard-container'><h2 style='text-align: center; color: black;'>Dashboard</h2></div>", unsafe_allow_html=True)
# dashboard.divider()
d1, d2 = dashboard.columns(2)

with d1:
    st.markdown("<h3 style='text-align: center; color: black;'>Options</h3>", unsafe_allow_html=True)
    input_type = st.radio(
        "Input type:",
        ["From File", "From Live Video Feed"],
        horizontal=True
    )
    st.button("Clear",on_click = clear)
    
with d2:
    inf_speed = st.session_state.inf_speed
    summary = st.session_state.summary
    st.markdown(
        f"""
        <div class='col1'>
        <h3 style='text-align: center; color: black;'>Detection Results</h3>
        <p>Inference Speed: {round(inf_speed, 2)} ms</p>
        <p>{summary[0] if len(summary) > 0 else ''}</p>
        </div>
        """, 
        unsafe_allow_html=True)