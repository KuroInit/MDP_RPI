import subprocess
import cv2

CAPTURED_IMAGE_PATH = "test_capture.jpg"


def capture_image_with_libcamera():
    try:
        cmd = [
            "libcamera-jpg",
            "-o",
            CAPTURED_IMAGE_PATH,
            "--width",
            "640", 
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
    if not capture_image_with_libcamera():
        print("Image capture failed.")
        return

    frame = cv2.imread(CAPTURED_IMAGE_PATH)
    if frame is None or frame.size == 0:
        print("Failed to load the captured image.")
        return

    print("Original image shape:", frame.shape)

    resized_frame = cv2.resize(frame, (640, 640))
    print("Resized image shape:", resized_frame.shape)

    cv2.imshow("Resized Image", resized_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_capture_and_resize()
