import time
import os
import subprocess
import psutil
import socket
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from web_server.utils.pathFinder import PathFinder
from web_server.utils.helper import commandGenerator
from config.logging_config import loggers  # Import Loguru config
from web_server.utils.imageRec import loadModel, predictImage
import cv2
from ultralytics import YOLO

# Use Picamera2 instead of the legacy PiCamera
from picamera2 import Picamera2

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="web_server/templates")
logger = loggers["webserver"]

# Define the socket path used for IPC with the STM service
STM_SOCKET_PATH = "/tmp/stm_ipc.sock"


# System resource analysis
def get_system_info():
    """Retrieve system resource usage and temperature."""
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    memory_used = memory.used / (1024**2)
    memory_total = memory.total / (1024**2)
    memory_percent = memory.percent
    uptime_seconds = time.time() - psutil.boot_time()
    uptime = time.strftime("%H:%M:%S", time.gmtime(uptime_seconds))

    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = int(f.read().strip()) / 1000.0
    except FileNotFoundError:
        temp = None  # Temp not available

    return {
        "cpu_usage": cpu_usage,
        "memory_used": round(memory_used, 2),
        "memory_total": round(memory_total, 2),
        "memory_percent": memory_percent,
        "uptime": uptime,
        "cpu_temp": round(temp, 2) if temp is not None else "N/A",
    }


@app.get("/wifi")
async def wifi_status():
    """Returns the current WiFi status of the Raspberry Pi."""
    try:
        result = subprocess.run(["iwconfig", "wlan0"], capture_output=True, text=True)
        return {"wifi_status": result.stdout}
    except Exception as e:
        logger.error(f"Error retrieving WiFi status: {e}")
        return {"error": "Failed to retrieve WiFi status"}


@app.get("/connected-devices")
async def connected_devices():
    """Lists devices connected to the Raspberry Pi hotspot."""
    try:
        result = subprocess.run(["arp", "-a"], capture_output=True, text=True)
        return {"connected_devices": result.stdout.split("\n")}
    except Exception as e:
        logger.error(f"Error retrieving connected devices: {e}")
        return {"error": "Failed to retrieve connected devices"}


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Returns an HTML page with real-time system stats."""
    system_info = get_system_info()
    return templates.TemplateResponse(
        "dashboard.html", {"request": request, "system": system_info}
    )


@app.get("/api/system-info")
async def system_info():
    """Returns system statistics in JSON format."""
    return get_system_info()


class PathFindingRequest(BaseModel):
    obstacles: list
    retrying: bool
    robot_x: int
    robot_y: int
    robot_dir: int
    big_turn: int = 0


@app.get("/")
def readRoot():
    return {"Backend": "Running"}


@app.get("/status")
async def status():
    """Health check endpoint."""
    return {"result": "ok"}


@app.get("/test-picam", response_class=HTMLResponse)
async def test_picam(request: Request):
    """
    Endpoint to test PiCamera capture, model inference, and display the output image
    with bounding boxes overlaid on the webserver.
    """
    try:
        # Initialize and configure Picamera2 for a still capture.
        picam2 = Picamera2()
        config = picam2.create_still_configuration()
        picam2.configure(config)
        picam2.start()

        # Define uploads directory and filename.
        uploads_dir = "uploads"
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
        timestamp = int(time.time())
        image_name = f"picam_{timestamp}.jpg"
        image_path = os.path.join(uploads_dir, image_name)

        # Capture and save the image.
        picam2.capture_file(image_path)
        picam2.stop()
        logger.info(f"Image captured and saved to {image_path}")

        # Load the ONNX model using Ultralytics YOLO.
        model_path = (
            "/home/mdp23/MDP_RPI/web_server/utils/trained_models/v8_white_bg.onnx"
        )
        model = YOLO(model_path)
        logger.info(f"Loaded Ultralytics YOLO model from {model_path}")

        # Run inference on the captured image.
        results = model(image_path)  # This returns a list of result objects.
        result = results[0]  # Use the first (and typically only) result.
        logger.info(f"Inference result: {result}")

        # Extract bounding boxes from the result.
        # Note: The structure of 'result' depends on your model.
        # Here we assume 'result.boxes' contains the bounding boxes.
        boxes = result.boxes if hasattr(result, "boxes") else []

        # Load the image using OpenCV.
        image = cv2.imread(image_path)

        # Annotate the image with bounding boxes.
        # Here we iterate over each detected box.
        for box in boxes:
            # box.xyxy contains the bounding box coordinates as a tensor.
            coords = box.xyxy[0].cpu().numpy().astype(int)  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = coords
            # Optionally, you can retrieve class info or confidence here.
            label = str(box.cls.cpu().numpy()[0]) if hasattr(box, "cls") else "object"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Save the annotated image.
        annotated_image_name = f"annotated_{image_name}"
        annotated_image_path = os.path.join(uploads_dir, annotated_image_name)
        cv2.imwrite(annotated_image_path, image)
        logger.info(f"Annotated image saved to {annotated_image_path}")

        # Encode the annotated image in base64 to embed it in HTML.
        with open(annotated_image_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode("utf-8")

        # Build the HTML content.
        html_content = f"""
        <html>
        <head>
            <title>PiCamera Test &amp; Inference</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                img {{ max-width: 100%; height: auto; }}
                pre {{ background: #f4f4f4; padding: 10px; }}
            </style>
        </head>
        <body>
            <h1>PiCamera Capture and Model Inference</h1>
            <h2>Annotated Image</h2>
            <img src="data:image/jpeg;base64,{encoded_image}" alt="Annotated Image" />
            <h2>Inference Results</h2>
            <pre>{result}</pre>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Error in /test-picam endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Test PiCamera error: {e}")


@app.post("/path")
async def pathFinding(request: PathFindingRequest, background_tasks: BackgroundTasks):
    """Main endpoint for the path finding algorithm."""
    content = request.model_dump()

    # Extract data from request
    obstacles = content["obstacles"]
    big_turn = int(content["big_turn"])
    retrying = content["retrying"]
    robot_x, robot_y = content["robot_x"], content["robot_y"]
    robot_direction = int(content["robot_dir"])

    maze_solver = PathFinder(20, 20, robot_x, robot_y, robot_direction, big_turn=None)

    # Add obstacles
    for ob in obstacles:
        maze_solver.add_obstacle(ob["x"], ob["y"], ob["d"], ob["id"])

    start = time.time()
    optimal_path, distance = maze_solver.get_optimal_order_dp(retrying=retrying)
    logger.info(
        f"Pathfinding time: {time.time() - start:.2f}s | Distance: {distance} units"
    )

    # Generate movement commands
    commands = commandGenerator(optimal_path, obstacles)

    # Process path results
    path_results = [optimal_path[0].get_dict()]
    i = 0
    for command in commands:
        if command.startswith(("SNAP", "FIN")):
            continue
        elif command.startswith(("FW", "FS", "BW", "BS")):
            i += int(command[2:]) // 10
        else:
            i += 1
        path_results.append(optimal_path[i].get_dict())

    result = {
        "data": {"distance": distance, "path": path_results, "commands": commands},
        "error": None,
    }
    # Automatically forward the result to run_task1 as a background task.
    background_tasks.add_task(run_task1, result)
    return result


# Define a Pydantic model for STM commands
class STMCommandRequest(BaseModel):
    command: str


def send_command_to_stm(command: str, socket_path: str = STM_SOCKET_PATH) -> str:
    """
    Connects to the STM IPC server via a Unix Domain Socket,
    sends the given command, and returns the response.
    """
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.connect(socket_path)
        client.send((command + "\n").encode("utf-8"))
        response = client.recv(1024)
        return response.decode("utf-8")
    except Exception as e:
        logger.error(f"Error communicating with STM: {e}")
        raise HTTPException(status_code=500, detail=f"STM communication error: {e}")
    finally:
        client.close()


@app.post("/send-stm-command")
async def send_stm_command(stm_command: STMCommandRequest):
    """
    Endpoint to send a command to the STM service via IPC.
    The command is forwarded to the STM service, which then processes it.
    """
    response = send_command_to_stm(stm_command.command)
    return {"stm_response": response}


def snap_handler(command: str):
    """
    Snap handler for SNAP commands.
    Extracts the numeric part (if any) and logs the snap command.
    Uses Picamera2 instead of the legacy PiCamera.
    """
    # Extract the number after "SNAP" (if any)
    num = command[4:].strip()
    loggers.info(f"Snap command received: {num}")

    try:
        # Initialize Picamera2
        picam2 = Picamera2()
        # Configure for still capture
        config = picam2.create_still_configuration()
        picam2.configure(config)
        picam2.start()

        img_name = f"snap_{int(time.time())}.jpg"
        img_path = os.path.join("uploads", img_name)
        # Capture image directly to a file
        picam2.capture_file(img_path)
        picam2.stop()
        loggers.info(f"Image saved: {img_path}")
    except Exception as e:
        loggers.error(f"Failed to capture image: {e}")
        return

    # Load the ONNX model
    session = loadModel()

    # Run inference
    result = predictImage(img_name, session)
    loggers.info(f"Inference result: {result}")

    return result


def run_task1(result: dict):
    commands = result.get("data", {}).get("commands", [])
    if not commands:
        logger.error("No commands found in algorithm result.")
        return

    for i, command in enumerate(commands):
        if command.upper().startswith("SNAP"):
            snap_handler(command)
        else:
            ack_received = False
            while not ack_received:
                try:
                    response = send_command_to_stm(command)
                    logger.info(f"Sent command: {command}, STM response: {response}")
                    if "ACK" in response:
                        ack_received = True
                    else:
                        logger.info("ACK not received yet; waiting...")
                        time.sleep(1)  # delay before checking again
                except Exception as e:
                    logger.error(f"Error sending command {command}: {e}")
                    time.sleep(1)  # wait before retrying on error

        # If there is another command to send, send the "ST00" command first and wait for ACK.
        if i < len(commands) - 1:
            ack_received = False
            while not ack_received:
                try:
                    response = send_command_to_stm("ST00")
                    logger.info(f"Sent command: ST00, STM response: {response}")
                    if "ACK" in response:
                        ack_received = True
                    else:
                        logger.info("ACK not received for ST00; waiting...")
                        time.sleep(1)
                except Exception as e:
                    logger.error(f"Error sending command ST00: {e}")
                    time.sleep(1)

        # Delay between commands; adjust as needed based on robot response time
        time.sleep(1)
    logger.info("Task1 execution complete")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)  # Change back to 0.0.0.0
