#!/usr/bin/env python3
import socketserver
import struct
import io
import os
import time

from PIL import Image
from picamera2 import Picamera2

SOCKET_PATH = "/tmp/camera.sock"


class CameraRequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        # Loop to continuously read commands until the client disconnects
        while True:
            command_line = self.rfile.readline().strip()
            if not command_line:
                # No data means the client has disconnected
                break
            command = command_line.decode("utf-8")
            print(f"Received command: {command}")

            if command == "SNAP00":
                # Capture an image as a numpy array using the warmed-up camera
                img_array = self.server.camera.capture_array("main")

                # Convert the image to JPEG in memory using Pillow
                img = Image.fromarray(img_array)
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_data = buffer.getvalue()

                # Send a 4-byte big-endian integer representing the length of the image data
                self.wfile.write(struct.pack(">I", len(jpeg_data)))
                # Then send the JPEG image data itself
                self.wfile.write(jpeg_data)
                self.wfile.flush()
                print(f"Sent image of size {len(jpeg_data)} bytes.")
            else:
                # Handle unknown commands by sending an error message
                err_msg = "Unknown command\n".encode("utf-8")
                self.wfile.write(struct.pack(">I", len(err_msg)))
                self.wfile.write(err_msg)
                self.wfile.flush()
                print("Sent error: Unknown command")
        print("Client disconnected.")


class ThreadedUnixStreamServer(
    socketserver.ThreadingMixIn, socketserver.UnixStreamServer
):
    pass


def main():
    # Remove existing socket file if it exists
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    # Initialize and configure Picamera2
    picam2 = Picamera2()
    config = picam2.create_still_configuration()
    picam2.configure(config)
    picam2.start()
    # Allow the sensor to warm up
    time.sleep(2)
    print("Camera warmed up and ready.")

    # Create the Unix domain socket server
    with ThreadedUnixStreamServer(SOCKET_PATH, CameraRequestHandler) as server:
        # Attach the camera instance to the server so that request handlers can use it
        server.camera = picam2
        print(
            f"Camera service started on Unix socket {SOCKET_PATH}. Waiting for commands..."
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Shutting down server.")
        finally:
            picam2.stop()
            server.server_close()
            os.remove(SOCKET_PATH)


if __name__ == "__main__":
    main()
