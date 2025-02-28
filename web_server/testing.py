import cv2
from ultralytics import YOLO
import onnx
import onnxruntime

MODEL_PATH = "web_server/utils/trained_models/v8_white_bg.onnx"
model = YOLO(MODEL_PATH)
img = cv2.imread("test.jpg")  # Use a known-good image file.
if img is None:
    print("Image load failed!")
else:
    results = model(img, verbose=False)
    print("Inference complete. Results:", results)
