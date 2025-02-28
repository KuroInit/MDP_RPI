import serial
import time
import numpy as np
import subprocess
from ultralytics import YOLO
from PIL import Image  # Use Pillow to load images

# Mapping of detection names to numeric values.
NAME_TO_CHARACTER = {
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

# Load the YOLO ONNX model locally.
MODEL_PATH = "/utils/trained_models/v8_white_bg.onnx"
model = YOLO(MODEL_PATH)

# Path to temporarily store captured image.
CAPTURED_IMAGE_PATH = "capture.jpg"


# Capture image using libcamera-jpeg with resolution 640x640.
def capture_image_with_libcamera_jpeg():
    try:
        # Build the libcamera-jpeg command with the desired resolution.
        command = f"libcamera-jpeg -o {CAPTURED_IMAGE_PATH} --width 640 --height 640"
        subprocess.run(command, shell=True, check=True)
        print("Image captured using libcamera-jpeg.")
        return True
    except Exception as e:
        print("Error capturing image with libcamera-jpeg:", e)
        return False


def send_command(command):
    """
    Sends a command to the robot via serial.
    """
    try:
        ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)  # Adjust port as needed.
        ser.flush()
        ser.write(command.encode())
        print(f"Sent: {command}")
        time.sleep(0.1)
        response = ser.readline().decode("utf-8").strip()
        print(f"Received: {response}")
        ser.close()
    except serial.SerialException as e:
        print(f"Serial error: {e}")


def capture_and_check():
    """
    Captures an image using libcamera-jpeg and runs inference using the local YOLO ONNX model.
    Returns True if the detected face is "Bullseye" (i.e. a valid face), otherwise False.
    """
    if not capture_image_with_libcamera_jpeg():
        return False

    try:
        # Load the image using Pillow and ensure it's in RGB format.
        image = Image.open(CAPTURED_IMAGE_PATH).convert("RGB")
        frame = np.array(image)
    except Exception as e:
        print("Failed to load the captured image:", e)
        return False

    # Verify the image is as expected.
    if frame is None or frame.size == 0:
        print("Failed to load the captured image or image is empty.")
        return False

    if frame.dtype != np.uint8:
        print(f"Captured frame has dtype {frame.dtype}; converting to uint8.")
        frame = frame.astype(np.uint8)

    # Run inference on the captured frame.
    results = model(frame, verbose=False)
    detected_character = "NA"
    for result in results:
        if (
            result.boxes is not None
            and result.boxes.conf is not None
            and len(result.boxes.conf) > 0
        ):
            conf_tensor = result.boxes.conf
            max_conf = float(conf_tensor.max())
            idx = int(conf_tensor.argmax())
            if max_conf >= 0.5:  # Confidence threshold; adjust as needed.
                best_result_id = int(result.boxes.cls[idx])
                # Map the detected numeric id back to a name.
                for key, val in NAME_TO_CHARACTER.items():
                    if val == best_result_id:
                        detected_character = key
                        break
    print("Detected character:", detected_character)
    return detected_character == "Bullseye"


def check_block_faces():
    """
    Checks each of the 4 faces of a square object.
    """
    valid_face_found = False

    for face in range(4):
        print(f"\nChecking face {face + 1}...")
        send_command("SF010:")  # Move forward 10 cm.
        time.sleep(1)

        if capture_and_check():
            print(f"Valid face found on face {face + 1}.")
            valid_face_found = True
            break
        else:
            print(f"Face {face + 1} is not valid.")
            send_command("RL090:")  # Turn left.
            time.sleep(1)
            send_command("SF010:")  # Move straight.
            time.sleep(1)
            send_command("RF090:")  # Turn right.
            time.sleep(1)
            send_command("RF090:")  # Turn right again.
            time.sleep(1)

    if not valid_face_found:
        print("No valid face found on any side.")
    return valid_face_found


if __name__ == "__main__":
    if check_block_faces():
        print("Robot has found a valid face. Stopping further movements.")
    else:
        print("Robot did not find a valid face.")
