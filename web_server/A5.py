import serial
from PIL import Image  # Use Pillow to load images
import onnx
import time
import os
import subprocess
import psutil
import socket
import glob
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from config.logging_config import loggers  # Import Loguru config
from web_server.utils.imageRec import loadModel, predictImage
from stm_comm.serial_comm import notify_bluetooth
import cv2
from ultralytics import YOLO
import onnxruntime
import numpy as np
import sys
from flask import jsonify

# Mapping of detection names to numeric values.
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

NAME_TO_CHARACTOR_ANDROID = {
    "One": "11",
    "Two": "12",
    "Three": "13",
    "Four": "14",
    "Five": "15",
    "Six": "16",
    "Seven": "17",
    "Eight": "18",
    "Nine": "19",
    "A": "20",
    "B": "21",
    "C": "22",
    "D": "23",
    "E": "24",
    "F": "25",
    "G": "26",
    "H": "27",
    "S": "28",
    "T": "29",
    "U": "30",
    "V": "31",
    "W": "32",
    "X": "33",
    "Y": "34",
    "Z": "35",
    "Up": "36",
    "Down": "37",
    "Right": "38",
    "Left": "39",
    "Stop": "40",
}


# Load the YOLO ONNX model locally.
MODEL_PATH = "utils/trained_models/v8_white_bg.onnx"
model = YOLO(MODEL_PATH)

# Path to temporarily store captured image.
CAPTURED_IMAGE_PATH = "capture.jpg"


# Capture image using libcamera-jpeg with resolution 640x640.
# def capture_image_with_libcamera_jpeg():
#     try:
#         # Build the libcamera-jpeg command with the desired resolution.
#         command = f"libcamera-jpeg -o {CAPTURED_IMAGE_PATH} --width 640 --height 640"
#         subprocess.run(command, shell=True, check=True)
#         print("Image captured using libcamera-jpeg.")
#         return True
#     except Exception as e:
#         print("Error capturing image with libcamera-jpeg:", e)
#         return False

def snap_handler():
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        best_conf = 0.0
        best_area = 0.0
        best_result = None
        best_result_charactor = "NA"
        best_frame_path = None

        try:
            picam2 = Picamera2()
            picam2.configure(picam2.create_still_configuration())
            config = picam2.create_still_configuration()
            picam2.configure(config)
            picam2.start()
            logger.info("Camera initialized.")
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return

        for i in range(3):
            frame_path = os.path.join(RESULT_IMAGE_DIR, f"SNAP{timestamp}_{i}.jpg")
            frame = picam2.capture_array()
            cv2.imwrite(frame_path, frame)
            logger.info(f"Captured image: {frame_path}")
            if frame is None or frame.size == 0:
                logger.error(f"Cannot load image captured: {frame_path}")
                continue

            if model is None:
                logger.error("Model not loaded; skipping inference.")
                continue

            results = model(frame, verbose=False)
            logger.info("Model inference completed.")

            # Iterate over all detections in each result
            for result in results:
                if (
                    result.boxes is not None
                    and result.boxes.conf is not None
                    and len(result.boxes.conf) > 0
                ):
                    for j in range(len(result.boxes.conf)):
                        conf_val = float(result.boxes.conf[j])
                        # Skip detections below the threshold
                        if conf_val < CONF_THRESHOLD:
                            continue

                        # Skip Bullseye detection (class ID 30)
                        if int(result.boxes.cls[j]) == 30:
                            continue

                        # Calculate bounding box area from xyxy coordinates
                        bbox = result.boxes.xyxy[j]
                        bbox = bbox.tolist() if hasattr(bbox, "tolist") else list(bbox)
                        x1, y1, x2, y2 = bbox
                        area = (x2 - x1) * (y2 - y1)

                        # If current detection has a higher confidence, update immediately.
                        if conf_val > best_conf:
                            best_conf = conf_val
                            best_area = area
                            best_result_id = int(result.boxes.cls[j])
                            best_result_charactor = list(NAME_TO_CHARACTOR.keys())[
                                list(NAME_TO_CHARACTOR.values()).index(best_result_id)
                            ]
                            best_result = result
                            best_frame_path = frame_path
                        # If the detection's confidence is within 10% of the best, compare the bounding box area.
                        elif (
                            best_conf > 0
                            and (conf_val / best_conf >= 0.9)
                            and (area > best_area)
                        ):
                            best_area = area
                            best_result_id = int(result.boxes.cls[j])
                            best_result_charactor = list(NAME_TO_CHARACTOR.keys())[
                                list(NAME_TO_CHARACTOR.values()).index(best_result_id)
                            ]
                            best_result = result
                            best_frame_path = frame_path

            time.sleep(1)

        picam2.close()

        # Save the annotated image
        result_image_path = os.path.join(RESULT_IMAGE_DIR, f"SNAPBEST{timestamp}.jpg")
        if best_result is not None and best_frame_path is not None:
            frame = cv2.imread(best_frame_path)
            annotated_frame = best_result.plot()
            cv2.imwrite(result_image_path, annotated_frame)
            logger.info(
                f"Detected ID: {best_result_charactor}, Confidence: {best_conf}, Saved to: {best_frame_path}, Save Path: {result_image_path}"
            )
        else:
            logger.info("No valid result found.")
            if best_frame_path is not None:
                frame = cv2.imread(best_frame_path)
                cv2.imwrite(result_image_path, frame)

        # Return the detected character (image id)
        return best_result_charactor

    except Exception as e:
        logger.error(f"Error in snap_handler: {e}")


def send_command(command):
    """
    Sends a command to the robot via serial.
    The function blocks and does not return until an "A" is received as the final response.
    """
    try:
        ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)  # Adjust port as needed.
        ser.flush()

        # Send the command to the robot.
        ser.write(command.encode())
        print(f"Sent: {command}")

        # Keep reading responses until we get an "A".
        while True:
            response = ser.readline().decode("utf-8").strip()
            if response == "A":
                print("Received final handshake 'A'.")
                break
            elif response:
                print(f"Ignored response: {response}")

        ser.close()
    except serial.SerialException as e:
        print(f"Serial error: {e}")


# def capture_and_check():
#     """
#     Captures an image using libcamera-jpeg and runs inference using the local YOLO ONNX model.
#     Returns True if the detected face is valid (i.e. not "Bullseye"), otherwise False.
#     """
#     if not capture_image_with_libcamera_jpeg():
#         return False

#     try:
#         # Load the image using Pillow and ensure it's in RGB format.
#         image = Image.open(CAPTURED_IMAGE_PATH).convert("RGB")
#         frame = np.array(image)
#     except Exception as e:
#         print("Failed to load the captured image:", e)
#         return False

#     # Verify the image is as expected.
#     if frame is None or frame.size == 0:
#         print("Failed to load the captured image or image is empty.")
#         return False

#     if frame.dtype != np.uint8:
#         print(f"Captured frame has dtype {frame.dtype}; converting to uint8.")
#         frame = frame.astype(np.uint8)

#     # Run inference on the captured frame.
#     results = model(frame, verbose=False)
#     detected_character = "NA"
#     for result in results:
#         if (
#             result.boxes is not None
#             and result.boxes.conf is not None
#             and len(result.boxes.conf) > 0
#         ):
#             conf_tensor = result.boxes.conf
#             max_conf = float(conf_tensor.max())
#             idx = int(conf_tensor.argmax())
#             if max_conf >= 0.5:  # Confidence threshold; adjust as needed.
#                 best_result_id = int(result.boxes.cls[idx])
#                 # Map the detected numeric id back to a name.
#                 for key, val in NAME_TO_CHARACTER.items():
#                     if val == best_result_id:
#                         detected_character = key
#                         break
#     print("Detected character:", detected_character)
#     # Valid face is any face that is not "Bullseye"
#     return detected_character != "Bullseye"

def startcarpark():
    send_command("UF100") # Forward till first 10x10 block
    time.sleep(0.5)
    target_id = snap_handler()
    target_id_android = NAME_TO_CHARACTOR_ANDROID.get(target_id, "NA")
    return target_id_android

def obstacle1_left():
    send_command("LF090")
    time.sleep(0.5)
    send_command("RF090")
    time.sleep(0.5)
    send_command("KF100") # RIGHT IR trace till edge of obstacle 1
    time.sleep(0.5)
    send_command("RF090")
    time.sleep(0.5)
    send_command("LF090")
    time.sleep(0.5)
    send_command("UF100")
    time.sleep(0.5)
    target_id = snap_handler()
    target_id_android = NAME_TO_CHARACTOR_ANDROID.get(target_id, "NA")
    return target_id_android    


def obstacle1_right():
    send_command("RF090")
    time.sleep(0.5)
    send_command("LF090")
    time.sleep(0.5)
    send_command("IF100") # LEFT IR trace till edge of obstacle 1
    time.sleep(0.5)
    send_command("LF090")
    time.sleep(0.5)
    send_command("RF090")
    time.sleep(0.5)
    send_command("UF100")
    time.sleep(0.5)
    target_id = snap_handler()
    target_id_android = NAME_TO_CHARACTOR_ANDROID.get(target_id, "NA")
    return target_id_android     

def obstacle2_left():
    Displacement = "000"
    send_command("LF090")
    time.sleep(0.5)
    # start tracking displacement here (replace line)
    # Displacement =
    send_command("KF100")
    time.sleep(0.5)
    send_command("RF090")
    time.sleep(0.5)
    send_command("RF090")
    time.sleep(0.5)
    send_command("KF100")
    time.sleep(0.5)
    send_command("RF090")
    time.sleep(0.5)
    send_command("RF090")
    time.sleep(0.5)
    send_command("SF000") # Displacement value (replace)
    time.sleep(0.5)
    send_command("LF090")
    time.sleep(0.5)
    send_command("UF100")
    return

def obstacle2_right():
    Displacement = "000"
    send_command("RF090")
    time.sleep(0.5)
    # start tracking displacement here (replace line)
    # Displacement =
    send_command("IF100")
    time.sleep(0.5)
    send_command("LF090")
    time.sleep(0.5)
    send_command("LF090")
    time.sleep(0.5)
    send_command("IF100")
    time.sleep(0.5)
    send_command("LF090")
    time.sleep(0.5)
    send_command("LF090")
    time.sleep(0.5)
    send_command("SF000") # Displacement value (replace)
    time.sleep(0.5)
    send_command("RF090")
    time.sleep(0.5)
    send_command("UF100")
    return

def endcapark():
    send_command("LF090")
    time.sleep(0.5)
    send_command("RF090")
    time.sleep(0.5)
    send_command("RF090")
    time.sleep(0.5)
    send_command("LF090")
    time.sleep(0.5)
    send_command("VF100")

def runtask2():
    """
    Code for task 2 here (yet to implement measure distance on stm)
    """
    Arrow1 = "NA"
    Arrow2 = "NA"

    Arrow1 = startcarpark()
    time.sleep(0.5)

    if Arrow1 == "39": # Obs 1 - Left Arrow
        Arrow2 = obstacle1_left
    elif Arrow1 == "38": # Obs 1 - Right Arrow
        Arrow2 = obstacle1_right
    else:
        print("Error executing carpark code")
    
    time.sleep(0.5)

    if Arrow2 == "39": # Obs 2 - Left Arrow
        obstacle2_left
    elif Arrow2 == "38": # Obs 2 - Right Arrow
        obstacle2_right
    else:
        print("Error executing obstacle 1 code")    

    time.sleep(0.5)
    endcapark
    return

if __name__ == "__main__":
    if runtask2():
        print("Task 2 completed.")
    else:
        print("Task 2 failed to run successfully.")