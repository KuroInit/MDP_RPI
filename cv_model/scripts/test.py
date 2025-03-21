import tkinter as tk
from tkinter import messagebox
import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk
import threading
import time

MODEL_PATH = "/Users/jz/Desktop/MDP_CV/v7_final.onnx"  

class VideoApp:

    def __init__(self, master):
        """
        Constructor: Initialize the UI elements and model loading.
        """
        self.master = master
        self.master.title("YOLO Detection Demo")
        self.master.geometry("900x700")
        self.model = YOLO(MODEL_PATH)

        # Creating UI elements
        self.create_widgets()

        # For the camera
        self.cap = None
        self.camera_thread = None
        self.is_running = False  # This flag controls the detection loop

    def create_widgets(self):
        """
        Create and place widgets (buttons, labels) in the main window.
        """
        # Video display label
        self.video_label = tk.Label(self.master, text="Video Feed")
        self.video_label.pack(pady=10)

        # Frame for control buttons
        btn_frame = tk.Frame(self.master)
        btn_frame.pack(pady=5)

        # Start detection button
        self.start_button = tk.Button(btn_frame, text="Start Detection", command=self.start_detection, width=15)
        self.start_button.grid(row=0, column=0, padx=5)

        # Stop detection button
        self.stop_button = tk.Button(btn_frame, text="Stop Detection", command=self.stop_detection, width=15)
        self.stop_button.grid(row=0, column=1, padx=5)

        # Quit button
        self.quit_button = tk.Button(btn_frame, text="Quit", command=self.quit_app, width=15)
        self.quit_button.grid(row=0, column=2, padx=5)

    def open_mac_camera(self):

        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Cannot access Mac camera!")
            return False
        return True

    def open_pi_camera(self):
        try:
            from picamera2 import Picamera2  
            self.cap = Picamera2()
            # self.cap.configure(self.cap.create_preview_configuration(main={"format": 'RGB888'}))
            self.cap.start()
            return True
        except Exception as e:
            messagebox.showerror("Camera Error", "Cannot access Raspberry Pi camera: " + str(e))
            return False


    def start_detection(self):
        if self.is_running:
            messagebox.showinfo("Info", "Detection is already running.")
            return

        # Use the selected camera type
        if self.camera_type.get() == "mac":
            if not self.open_mac_camera():
                return
        elif self.camera_type.get() == "pi":
            if not self.open_pi_camera():
                return

        self.is_running = True
        self.camera_thread = threading.Thread(target=self.run_detection_loop, daemon=True)
        self.camera_thread.start()

    def run_detection_loop(self):
        """
        Continuously read frames from the camera and apply YOLO detection.
        """
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret or frame is None:
                continue

            edges = self.apply_canny(frame)
            fused_frame = cv2.addWeighted(frame, 0.8, edges, 0.2, 0)

            # 1. Apply YOLOv8 model on the frame
            results = self.model(fused_frame, verbose=False)

            # 2. Visualize results using ultralytics' built-in function
            annotated_frame = results[0].plot()  # Returns an annotated numpy array (BGR)

            # 3. Convert BGR to RGB for Tk display
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=pil_image)

            # 4. Update UI in a thread-safe way: use master.after()
            self.master.after(0, self.update_video_label, imgtk)

            # 5. Sleep a bit or adjust for performance
            time.sleep(0.01)

    def update_video_label(self, imgtk):
        """
        Update the Label widget with the new frame.
        """
        self.video_label.config(image=imgtk)
        self.video_label.imgtk = imgtk  

    def stop_detection(self):

        if not self.is_running:
            messagebox.showinfo("Info", "Detection is already stopped.")
            return

        self.is_running = False

        # Release camera resource
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        self.video_label.config(image='')  # Clear the video display

    def quit_app(self):

        self.stop_detection()
        self.master.destroy()

    def apply_canny(self, image, threshold1=30, threshold2=100):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1, threshold2)
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        return edges_color


    def create_widgets(self):

        # Video display label
        self.video_label = tk.Label(self.master, text="Video Feed")
        self.video_label.pack(pady=10)

        # Add camera selection radio buttons
        self.camera_type = tk.StringVar(value="mac")
        radio_frame = tk.Frame(self.master)
        radio_frame.pack(pady=5)
        tk.Radiobutton(radio_frame, text="MacBook Camera", variable=self.camera_type, value="mac").pack(side="left", padx=5)
        tk.Radiobutton(radio_frame, text="Raspberry Pi Camera", variable=self.camera_type, value="pi").pack(side="left", padx=5)

        # Frame for control buttons
        btn_frame = tk.Frame(self.master)
        btn_frame.pack(pady=5)

        # Start detection button
        self.start_button = tk.Button(btn_frame, text="Start Detection", command=self.start_detection, width=15)
        self.start_button.grid(row=0, column=0, padx=5)

        # Stop detection button
        self.stop_button = tk.Button(btn_frame, text="Stop Detection", command=self.stop_detection, width=15)
        self.stop_button.grid(row=0, column=1, padx=5)

        # Quit button
        self.quit_button = tk.Button(btn_frame, text="Quit", command=self.quit_app, width=15)
        self.quit_button.grid(row=0, column=2, padx=5)


def main():

    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
