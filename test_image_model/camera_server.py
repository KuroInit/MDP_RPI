import os
import sys
import cv2
import time
import numpy as np
from flask import Flask, jsonify, request
import onnxruntime  # This import remains if you need onnxruntime elsewhere

try:
    from picamera2 import PiCamera2

    picamera_available = True
except ImportError:
    picamera_available = False

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
        # We'll use a local flag to determine which camera method to use.
        use_picamera = False

        if picamera_available:
            try:
                camera = PiCamera2()
                camera.resolution = (640, 480)
                time.sleep(2)  # Allow the camera to warm up
                use_picamera = True
            except Exception as e:
                app.logger.error(f"PiCamera initialization failed: {e}")
                # Fallback to OpenCV's VideoCapture if PiCamera fails
                camera = cv2.VideoCapture(0)
                if not camera.isOpened():
                    return jsonify({"error": "Camera Error: Cannot access camera"}), 500
        else:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                return jsonify({"error": "Camera Error: Cannot access camera"}), 500

        best_conf = 0.0
        best_result_id = None
        best_result = None
        best_result_charactor = "NA"  # Initialize with default value
        last_frame = None  # Track the last valid frame

        for i in range(10):
            # Capture frame using the selected method.
            if use_picamera:
                frame = np.empty((480, 640, 3), dtype=np.uint8)
                camera.capture(frame, "bgr")
            else:
                ret, frame = camera.read()
                if not ret or frame is None:
                    app.logger.warning(
                        "Failed to capture frame from camera. Skipping iteration."
                    )
                    continue

            # Validate and adjust image format:
            if frame is not None:
                # Check if the frame is grayscale; if so, convert to BGR.
                if len(frame.shape) == 2:
                    app.logger.warning(
                        "Captured frame is grayscale. Converting to BGR."
                    )
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

            # Ensure the frame is valid (non-empty).
            if frame is None or frame.size == 0:
                app.logger.error(
                    "Captured frame is empty or invalid. Skipping iteration."
                )
                continue

            # Save the current valid frame.
            last_frame = frame

            # Now pass the valid frame to the model.
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
                        # Map id to character using the NAME_TO_CHARACTOR dictionary.
                        best_result_charactor = list(NAME_TO_CHARACTOR.keys())[
                            list(NAME_TO_CHARACTOR.values()).index(best_result_id)
                        ]
                        best_result = result

            time.sleep(0.1)

        # If no valid frame was captured during the iterations, return an error.
        if last_frame is None:
            if use_picamera:
                camera.close()
            else:
                camera.release()
            return jsonify({"error": "No valid frame captured from the camera."}), 500

        # Save the output image.
        if best_result is not None:
            annotated_frame = best_result.plot()
            cv2.imwrite(OUTPUT_IMAGE_PATH, annotated_frame)
        else:
            cv2.imwrite(OUTPUT_IMAGE_PATH, last_frame)

        # Close the camera appropriately.
        if use_picamera:
            camera.close()
        else:
            camera.release()

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
