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
from web_server.utils.pathFinder import PathFinder
from web_server.utils.helper import commandGenerator
from config.logging_config import loggers  # Import Loguru config
from web_server.utils.imageRec import loadModel, predictImage
from stm_comm.serial_comm import notify_bluetooth
import cv2
from ultralytics import YOLO
import onnxruntime
import numpy as np
import sys
from flask import jsonify


# Use Picamera2 instead of the legacy PiCamera
from picamera2 import Picamera2

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

# Define confidence threshold
CONF_THRESHOLD = 0.4

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

RESULT_IMAGE_DIR = os.path.join(os.getcwd(), "web_server", "result_image")
os.makedirs(RESULT_IMAGE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(os.getcwd(), "web_server", "utils", "trained_models", "v9_noise_bg.onnx")
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    logger.error(f"Loading Model Failed: {e}")
    model = None


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
        elif command.startswith(("SF", "SB")):
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


def send_target_identification(obstacle_number: str, target_id: str):
    command = f"{obstacle_number},{target_id}"
    notify_bluetooth(command, 2)


def snap_handler(command: str, obid: str):

    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        best_conf = 0.0
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
            
            frame_path = os.path.join(RESULT_IMAGE_DIR, f"SNAP{obid}_{i}.jpg")
            frame = picam2.capture_array()
            cv2.imwrite(frame_path, frame)
            logger.info(f"Captured image: {frame_path}")
            if frame is None or frame.size == 0:

                logger.error(f"Cannot load image captured : {frame_path}")
                continue

            if model is None:
                logger.error("Model not loaded; skipping inference.")
                continue

            results = model(frame, verbose=False)
            logger.info(f"Model inference")  

            for result in results:
                if result.boxes is not None and result.boxes.conf is not None and len(result.boxes.conf) > 0:
                    if result.boxes.cls[idx] == 30:
                        continue # Skip Bullseye detection
                    conf_tensor = result.boxes.conf
                    max_conf = float(conf_tensor.max())
                    idx = int(conf_tensor.argmax())

                    if max_conf > best_conf and max_conf >= CONF_THRESHOLD:
                        best_conf = max_conf
                        best_result_id = int(result.boxes.cls[idx])
                        best_result_charactor = list(NAME_TO_CHARACTOR.keys())[
                            list(NAME_TO_CHARACTOR.values()).index(best_result_id)
                        ]
                        best_result = result
                        best_frame_path = frame_path

            time.sleep(1) 

        picam2.close()

        # Save the annotated image
        result_image_path = os.path.join(RESULT_IMAGE_DIR, f"SNAPBEST{obid}.jpg")
        if best_result is not None and best_frame_path is not None:
            frame = cv2.imread(best_frame_path)
            annotated_frame = best_result.plot()
            cv2.imwrite(result_image_path, annotated_frame)
            logger.info(f"Detected ID: {best_result_charactor}, Confidence: {best_conf}, Saved to: {best_frame_path}, Save Path: {result_image_path}")
        else:
            logger.info("No valid result found.")
            if best_frame_path is not None:
                frame = cv2.imread(best_frame_path)
                cv2.imwrite(result_image_path, frame)

        #return iamge id        
        return best_result_charactor
   
    except Exception as e:
        logger.error(f"Error in snap_handler: {e}")

def stitchImage():

    imgFolder = RESULT_IMAGE_DIR  # Directory where SNAP images are stored
    stitchedPath = os.path.join(imgFolder, 'SNAPALLSTITCHED.jpg')
    
    # Get paths of all SNAP images (SNAPBEST{obid}.jpg)
    imgPaths = glob.glob(os.path.join(imgFolder, "SNAPBEST*.jpg"))
    
    # Check if there are any images to stitch
    if not imgPaths:
        print("No SNAP images found to stitch.")
        return

    images = [cv2.imread(x) for x in imgPaths]
    
    # Check if all images were successfully loaded
    if any(img is None for img in images):
        print("Error: One or more images could not be loaded.")
        return

    # Calculate total width and max height for the stitched image
    total_width = sum(img.shape[1] for img in images)
    max_height = max(img.shape[0] for img in images)
    
    # Create a new blank image with the calculated dimensions
    stitchedImg = 255 * np.ones(shape=[max_height, total_width, 3], dtype=np.uint8)  # White background

    # Paste each image into the stitched image
    x_offset = 0
    for img in images:
        stitchedImg[:img.shape[0], x_offset:x_offset + img.shape[1]] = img
        x_offset += img.shape[1]
    
    # Save the stitched image using OpenCV
    cv2.imwrite(stitchedPath, stitchedImg)
   
    print(f"Stitched image saved to: {stitchedPath}")



def run_task1(result: dict):
    commands = result.get("data", {}).get("commands", [])
    if not commands:
        logger.error("No commands found in algorithm result.")
        return

    for command in commands:
        if command.upper().startswith("SNAP"):
            ob_id = command[4:]
            target_id = snap_handler(command, ob_id)
            target_id_android = NAME_TO_CHARACTOR_ANDROID.get(target_id, "NA")  # Default to NA if not found
            send_target_identification(ob_id, target_id_android) #send ob id and target id to bluetooth
        elif (command == "FIN"):
            stitchImage()
            notify_bluetooth("FIN", 1)
            break 
        else:
            ack_received = False
            while not ack_received:
                try:
                    response = send_command_to_stm(command)
                    logger.info(f"Sent command: {command}, STM response: {response}")
                    if "A" in response:
                        ack_received = True
                    else:
                        logger.info("ACK not received yet; waiting...")
                        time.sleep(0.5)  # Delay before checking again
                except Exception as e:
                    logger.error(f"Error sending command {command}: {e}")
                    time.sleep(0.5)  # Wait before retrying on error

        # Delay between commands; adjust as needed based on robot response time.
        time.sleep(0.5)

    logger.info("Task1 execution complete")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
