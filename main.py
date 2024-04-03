# Imports
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import detector as dt
import shutil
import re


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
    fourcc = cv2.VideoWriter_fourcc(*'X264')  # libx264 codec (mp4v does not work with streamlit)
    out = cv2.VideoWriter('tmp/output.mp4', fourcc, fps, (width, height))
    
    # Detect defects in the video
    results = detector.detect(video_path, input_type="video")
    inf_speed = 0
    
    # Process each frame and add to the output video
    for result in results:
        # Convert PIL Image to numpy array
        processed_frame = np.array(result.plot())
        # Add frame inference time to total inference time, add summary.verbose() to results
        inf_speed += result.speed['inference']
        st.session_state.summary.append(result.verbose())
        # Write the processed frame to output video
        out.write(processed_frame)
    
    # Release everything when job is finished
    cap.release()
    out.release()
    
    # Return inference speed
    return inf_speed


def remove_tmp(tmp_directory):
    """Remove tmp directory"""
    
    try:
        shutil.rmtree(tmp_directory)
        print(f"Directory '{tmp_directory}' has been removed successfully.")
    except FileNotFoundError:
        print(f"Directory '{tmp_directory}' does not exist.")
    except Exception as e:
        print(f"Error occurred while deleting directory '{tmp_directory}': {e}")


def clear():
    """Clear session state variables to reset program"""
    
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

## ---- UI ----


st.title("Defect Detector")
col1, col2 = st.columns(2)

# Set state variables
if 'in_file' not in st.session_state:
    st.session_state['in_file'] = None
if 'out_file' not in st.session_state:
    st.session_state['out_file'] = None
if 'inf_speed' not in st.session_state:
    st.session_state['inf_speed'] = 0
if 'summary' not in st.session_state:
    st.session_state['summary'] = []

# Input Column
with col1:
    st.header("Input")
    
    types = ["jpg", "jpeg", "png","mp4","mov","wmv"]
    file = st.session_state.in_file
    
    # Display File Uploader object if no file is selected
    if file is None:
        file = st.file_uploader("Choose an Image", type=types, key='file_uploader')
        st.session_state.in_file = file
        if file is not None:
            st.rerun()
    
    # Display the uploaded image/video when file is selected
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


# Output Column
with col2:
    st.header("Output")
    
    # Make Prediction and display spinner while processing
    if file is not None: 
        # Display spinner only when processing is required
        if st.session_state.out_file is None:
            with st.spinner(text="Detection in progress..."):
                if in_type == "image":
                    # Process image and update session state
                    output = dtr.detect(input, input_type=in_type)
                    # Display the processed image
                    for result in output:
                        st.session_state.out_file = result
                        st.session_state.inf_speed += result.speed['inference']
                        image = result.plot()
                        st.session_state.summary.append(result.verbose()) 
                elif in_type == "video":
                    # Process video and update session state
                    st.session_state.inf_speed = process_video(input, dtr)
                    with open("tmp/output.mp4", "rb") as file:
                        st.session_state.out_file = file.read()
                    remove_tmp('tmp')

        # Display outfile to the Output section of UI
        if st.session_state.out_file is not None:
            if in_type == "image":
                # Assuming st.session_state.out_file holds the processed image data
                st.image(st.session_state.out_file.plot(), caption="Output", use_column_width=True)
            elif in_type == "video":
                # Assuming st.session_state.out_file holds the binary video data
                st.video(st.session_state.out_file, format="video/mp4")
    else:
        # When no file is selected
        st.write("Result will show here.")
        st.session_state.inf_speed = 0
        st.session_state.summary = []


# Create Dashboard section at bottom
dashboard = st.container(border=True)

# Load CSS from style.css
css = open('style.css')
style = f"""
<style>
    {css.read()}
</style>
"""

# Create Dashboard header
dashboard.markdown(style, unsafe_allow_html=True)
dashboard.markdown("<div class='dashboard-container'><h2 style='text-align: center; color: black;'>Dashboard</h2></div>", unsafe_allow_html=True)

# Create dashboard columns
d1, d2 = dashboard.columns(2)

# Option menu
with d1:
    st.markdown("<h3 style='text-align: center; color: black;'>Options</h3>", unsafe_allow_html=True)
    
    # Radio selector to choose whether input comes from file or live feed
    input_type = st.radio(
        "Input type:",
        ["From File", "From Live Video Feed"],
        horizontal=True,
        disabled = True # Disabled since live video feed is not supported yet
    )
    
    # Clear button
    st.button("Clear",on_click = clear)
    
# Display Detection Results
with d2:
    inf_speed = st.session_state.inf_speed
    summary = st.session_state.summary
    summary = summary[len(summary)-1] if len(summary) > 0 else ''
    total_defects = sum(int(number) for number in re.findall(r'\d+', summary))
    defects = f'<p>Total Number of Defects: {total_defects}</p>' if len(summary) > 0 else ''
    st.markdown(
        f"""
        <div class='col1'>
        <h3 style='text-align: center; color: black;'>Detection Results</h3>
        <p>Inference Speed: {round(inf_speed, 2)} ms</p>
        {defects}
        <p>{summary}</p>
        </div>
        """, 
        unsafe_allow_html=True)