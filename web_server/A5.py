import serial
import time
import requests


def send_command(command):
    """
    Sends a command to the robot via serial.
    """
    try:
        ser = serial.Serial(
            "/dev/ttyUSB0", 115200, timeout=1
        )  # Adjust the port as needed.
        ser.flush()
        ser.write(command.encode())  # Send the command.
        print(f"Sent: {command}")
        time.sleep(0.1)  # Allow processing time.
        response = ser.readline().decode("utf-8").strip()
        print(f"Received: {response}")
        ser.close()
    except serial.SerialException as e:
        print(f"Serial error: {e}")


def capture_and_check():
    """
    Captures an image via a remote server (which runs YOLO ONNX inference)
    and checks if the detected character is "Bullseye" (mapped to 30).
    Returns True if a valid face is detected, otherwise False.
    """
    server_url = (
        "http://<Server IP>:5000/capture"  # Replace <Server IP> with your server's IP.
    )
    try:
        response = requests.get(server_url, timeout=5)
        data = response.json()
        print("Result from server:", data)
        # The server is assumed to return a JSON with key "result_id".
        detected_character = data.get("result_id", "NA")

        # Mapping of names to numeric values.
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

        # Check if the detected face is a Bullseye.
        if NAME_TO_CHARACTER.get(detected_character, -1) == 30:
            return True
        else:
            return False
    except Exception as e:
        print("Request failed:", e)
        return False


def check_block_faces():
    """
    Checks each of the 4 faces of a block.
    For each face:
      - The robot moves forward to approach the face.
      - The robot checks if the face is valid.
      - If not, it rotates 90° to check the next face.
    """
    valid_face_found = False

    for face in range(4):
        print(f"\nChecking face {face + 1}...")
        # Move forward 10 cm to check the current face.
        send_command("SF010:")
        time.sleep(1)  # Wait for movement completion and image capture.

        # Capture image and check if the current face is valid.
        if capture_and_check():
            print(f"Valid face found on face {face + 1}.")
            valid_face_found = True
            break
        else:
            print(f"Face {face + 1} is not valid.")
            # Rotate 90° to check the next face.
            send_command("RF090:")
            time.sleep(1)  # Wait for rotation to complete.

    if not valid_face_found:
        print("No valid face found on any side.")
    return valid_face_found


if __name__ == "__main__":
    # Run the routine to check block faces.
    if check_block_faces():
        print("Robot has found a valid face. Stopping further movements.")
    else:
        print("Robot did not find a valid face.")
