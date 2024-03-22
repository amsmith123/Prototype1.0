from ultralytics import YOLO
from PIL import Image

class Detector:
    # Initialize model
    def __init__(self,model):
        self.model = YOLO(model)
        # self.model = YOLO("NEUSSD\\yolov8l-seg.pt")
        
    # Infer on input. Works with image, video, or stream.
    def detect(self,source,input_type="image",task='detect'):
        stream=False
        if input_type.lower == "video" or input_type.lower == "v":
            stream=True
        results = self.model(source,stream=stream,task=task,classes=range(0,6))
        return results
        