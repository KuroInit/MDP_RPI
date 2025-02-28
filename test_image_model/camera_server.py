import os
import sys
import cv2
import time
import numpy as np
from flask import Flask, jsonify, request
import subprocess  # To run libcamera commands
import onnxruntime  # This import remains if you need onnxruntime elsewhere

from ultralytics import YOLO

NAME_TO_CHARACTOR = {
    "NA": "NA",
    "Bullseye": 30,
    "One": 0,
    "Two": 1,
    "Three": 2,
    "Four": 3,
    "Five": 4,
    "Six": 5,
    "Seven": 6,
    "Eight": 7,
    "Nine": 8,
    "A": 9,
    "B": 10,
    "C": 11,
    "D": 12,
    "E": 13,
    "F": 14,
    "G": 15,
    "H": 16,
    "S": 17,
    "T": 18,
    "U": 19,
    "V": 20,
    "W": 21,
    "X": 22,
    "Y": 23,
    "Z": 24,
    "Up": 25,
    "Down": 26,
    "Right": 27,
    "Left": 28,
    "Stop": 29,
}

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    CURRENT_DIR, "..", "web_server", "utils", "trained_models", "v8_white_bg.onnx"
)

# Save the output image in the current working directory
OUTPUT_IMAGE_PATH = os.path.join(os.getcwd(), "result.jpg")

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"[ERROR] Loading Model Failed: {e}", file=sys.stderr)
    model = None

CONF_THRESHOLD = 0.4

app = Flask(__name__)


@app.route("/capture", methods=["GET"])
def capture_and_detect():
    try:
        # Use libcamera for capture.
        use_libcamera = True

        if use_libcamera:
            # Capture image using libcamera-still with the specified command.
            # This command will capture an image with a 1000ms timeout and save it to OUTPUT_IMAGE_PATH.
            cmd = ["libcamera-still", "-o", OUTPUT_IMAGE_PATH, "--timeout", "1000"]
            subprocess.run(cmd, check=True)
            # Load the captured image.
            frame = cv2.imread(OUTPUT_IMAGE_PATH)
            if frame is None or frame.size == 0:
                app.logger.error("Failed to load image captured by libcamera-still.")
                return (
                    jsonify({"error": "Failed to capture image using libcamera-still"}),
                    500,
                )
        else:
            # Fallback method (using OpenCV's VideoCapture)
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return jsonify({"error": "Camera Error: Cannot access camera"}), 500
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                return jsonify({"error": "Failed to capture image from camera"}), 500

        # Validate and adjust image format.
        if frame is not None:
            # If the image is grayscale, convert to BGR.
            if len(frame.shape) == 2:
                app.logger.warning("Captured frame is grayscale. Converting to BGR.")
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif len(frame.shape) == 3:
                if frame.shape[2] == 4:
                    app.logger.warning(
                        "Captured frame has 4 channels. Converting to BGR."
                    )
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                elif frame.shape[2] != 3:
                    app.logger.warning(
                        f"Captured frame has {frame.shape[2]} channels. Truncating to 3 channels."
                    )
                    frame = frame[:, :, :3]
            if frame.dtype != np.uint8:
                app.logger.warning(
                    f"Captured frame has dtype {frame.dtype}. Converting to uint8."
                )
                frame = frame.astype(np.uint8)

        # Now run inference on the validated frame.
        best_conf = 0.0
        best_result = None
        best_result_charactor = "NA"

        results = model(frame, verbose=False)
        for result in results:
            if (
                result.boxes is not None
                and result.boxes.conf is not None
                and len(result.boxes.conf) > 0
            ):
                conf_tensor = result.boxes.conf
                max_conf = float(conf_tensor.max())
                idx = int(conf_tensor.argmax())
                if max_conf > best_conf and max_conf >= CONF_THRESHOLD:
                    best_conf = max_conf
                    best_result_id = int(result.boxes.cls[idx])
                    best_result_charactor = list(NAME_TO_CHARACTOR.keys())[
                        list(NAME_TO_CHARACTOR.values()).index(best_result_id)
                    ]
                    best_result = result

        # Save the output image.
        if best_result is not None:
            annotated_frame = best_result.plot()
            cv2.imwrite(OUTPUT_IMAGE_PATH, annotated_frame)
        else:
            cv2.imwrite(OUTPUT_IMAGE_PATH, frame)

        response = {
            "result_id": best_result_charactor,
            "probability": best_conf,
            "result_image_path": OUTPUT_IMAGE_PATH,
        }
        return jsonify(response), 200

    except Exception as e:
        app.logger.error(f"Handle request error: {str(e)}")
        return jsonify({"error": f"Server Error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888, debug=False)
