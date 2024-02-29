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
    #else:
        #st.write("Upload an image to get started")

with col2:
    st.header("Output")
    
    # Make Prediction
    if file is not None:
        st.write("predicting")
        results = dtr.detect(image)
        st.write('\n\n\n ')
        # out_image = Image(results)
        # st.image(out_image,caption="Output", use_column_width=True)
        
        for result in results:
            #print(result)
            boxes = result.boxes
            probs = result.probs
            # result.show()
            # out_image = Image(result)
            image = result.plot()
            st.image(image,caption="Output", use_column_width=True)
    else:
        st.write("Result will show here.")
        
col3, col4 = st.columns(2)

with col3:
    if st.button("Clear"):
        file = None

with col4:
    st.write("Probabilities:")
    if probs is not None:
        st.write(probs.top5)
    else:
        st.write("no Probability available.")
    


