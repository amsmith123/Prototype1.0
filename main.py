import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import detector as dt
import matplotlib.pyplot as plt 


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
    
    file = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"],key=10101)
    
    # Display the uploaded image
    if file is not None:
        image = Image.open(file)
        st.image(image, caption="Input", use_column_width=True)
    else:
        image = None
    # if st.button("Clear"):
    #     file = None

with col2:
    st.header("Output")
    for _ in range(0,9):
        st.write("")
    
    # Make Prediction
    if file is not None:
        st.write("predicting")
        results = dtr.detect(image)
        
        for result in results:
            speed = result.speed
            image = result.plot()
            st.image(image,caption="Output", use_column_width=True)
            st.write(f"Inference Speed: {speed['inference']} ms")
            st.write(result.verbose())
    else:
        st.write("Result will show here.")
        
    


