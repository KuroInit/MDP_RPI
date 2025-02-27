from flask import Flask, render_template, jsonify
import time
import os
from picamera2 import Picamera2
from web_server.utils.imageRec import loadModel, predictImage
from config.logging_config import loggers  # Import Loguru config

app = Flask(__name__)

def snap_handler():
    """
    Snap handler for capturing an image and running inference.
    """
    loggers.info("Snap command received.")

    try:
        # Initialize Picamera2
        picam2 = Picamera2()
        config = picam2.create_still_configuration()
        picam2.configure(config)
        picam2.start()

        img_name = f"snap_{int(time.time())}.jpg"
        img_path = os.path.join("static/uploads", img_name)
        os.makedirs("static/uploads", exist_ok=True)

        # Capture image
        picam2.capture_file(img_path)
        picam2.stop()
        loggers.info(f"Image saved: {img_path}")

        # Load the model and predict
        session = loadModel()
        result = predictImage(img_name, session)

        loggers.info(f"Inference result: {result}")
        return {"success": True, "image": img_path, "result": result}

    except Exception as e:
        loggers.error(f"Failed to capture image: {e}")
        return {"success": False, "error": str(e)}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/snap", methods=["POST"])
def snap():
    return jsonify(snap_handler())

if __name__ == "__main__":
    app.run(debug=True)