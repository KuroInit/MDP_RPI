import subprocess
import cv2

CAPTURED_IMAGE_PATH = "test_capture.jpg"

def capture_image():
    try:
        # Capture image using libcamera-jpg with a 1000ms timeout.
        cmd = ["libcamera-jpg", "-o", CAPTURED_IMAGE_PATH, "--timeout", "1000"]
        subprocess.run(cmd, check=True)
        print("Image captured using libcamera-jpg.")
        return True
    except Exception as e:
        print("Error capturing image:", e)
        return False

if __name__ == "__main__":
    if capture_image():
        # Load the captured image using OpenCV.
        img = cv2.imread(CAPTURED_IMAGE_PATH)
        if img is None or img.size == 0:
            print("Failed to load captured image.")
        else:
            print("Image loaded successfully.")
            print("Image shape:", img.shape)
            # Display the image (if you are in an environment that supports a display)
