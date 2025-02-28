import subprocess
import cv2

# Path to store the captured image.
CAPTURED_IMAGE_PATH = "test_capture.jpg"


def capture_image_with_libcamera():
    """
    Captures an image using libcamera-jpg with the given width/height.
    """
    try:
        cmd = [
            "libcamera-jpg",
            "-o",
            CAPTURED_IMAGE_PATH,
            "--width",
            "640",  # Optionally, set to your desired output size.
            "--height",
            "640",
            "--timeout",
            "1000",
        ]
        subprocess.run(cmd, check=True)
        print("Image captured using libcamera-jpg.")
        return True
    except Exception as e:
        print("Error capturing image with libcamera-jpg:", e)
        return False


def test_capture_and_resize():
    """
    Captures an image using libcamera-jpg, loads it with OpenCV,
    resizes it to 640x640, and displays the original and resized image shapes.
    """
    if not capture_image_with_libcamera():
        print("Image capture failed.")
        return

    # Load the captured image.
    frame = cv2.imread(CAPTURED_IMAGE_PATH)
    if frame is None or frame.size == 0:
        print("Failed to load the captured image.")
        return

    # Print original dimensions.
    print("Original image shape:", frame.shape)

    # Resize the image to 640x640.
    resized_frame = cv2.resize(frame, (640, 640))
    print("Resized image shape:", resized_frame.shape)

    # Display the resized image.
    cv2.imshow("Resized Image", resized_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_capture_and_resize()
