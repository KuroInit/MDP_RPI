import bluetooth
import os
import sys
import json
import requests
from config.logging_config import loggers

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Web server
ALGO_API_URL = "http://localhost:8000/path"

obstacles_list = []
robot_position = {"x": 1, "y": 1, "dir": 0}
direction_map = {"NORTH": 0, "EAST": 2, "SOUTH": 4, "WEST": 6}


def start_bluetooth_service():
    logger = loggers["bluetooth"]
    server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    server_sock.bind(("", bluetooth.PORT_ANY))
    server_sock.listen(1)
    port = server_sock.getsockname()[1]
    logger.info(f"Bluetooth listening on RFCOMM port {port}")

    bluetooth.advertise_service(
        server_sock,
        "RobotProjectBTService",
        service_id="00001101-0000-1000-8000-00805F9B34FB",
        service_classes=["00001101-0000-1000-8000-00805F9B34FB"],
        profiles=[bluetooth.SERIAL_PORT_PROFILE],
    )
    logger.info("Bluetooth service advertised.")

    try:
        while True:
            client_sock, client_info = server_sock.accept()
            logger.info(f"Accepted connection from {client_info}")
            handle_client_session(client_sock, logger)
            logger.info(f"Client {client_info} disconnected.")
    except Exception as e:
        logger.error(f"Bluetooth server error: {e}")
    finally:
        server_sock.close()
        logger.info("Bluetooth server socket closed.")


def handle_client_session(client_sock, logger):
    while True:
        try:
            data = client_sock.recv(1024)
            if not data:
                logger.info("Client closed the connection.")
                break
            message_str = data.decode("utf-8").strip()
            logger.info(f"Received: {message_str}")
            process_message(message_str, client_sock, logger)
        except Exception as e:
            logger.error(f"Error in client communication: {e}")
            break
    client_sock.close()


def process_message(message_str, client_sock, logger):
    parts = message_str.split(",")
    if not parts:
        logger.warning("Empty message parts.")
        return

    msg_type = parts[0].upper()
    if msg_type == "ROBOT":
        handle_robot(parts, logger)
    elif msg_type == "TARGET":
        handle_target(parts, logger)
    elif msg_type == "STATUS":
        handle_status(parts, logger)
    elif msg_type == "OBSTACLE":
        handle_obstacle(parts, logger)
    elif msg_type == "FACE":
        handle_face(parts, logger)
    elif msg_type == "MOVE":
        handle_move(parts, logger)
    elif msg_type == "CMD":
        handle_cmd(parts, logger, client_sock)
    else:
        logger.warning(f"Unknown message type: {msg_type}")


def handle_robot(parts, logger):
    global robot_position
    if len(parts) < 4:
        logger.error("ROBOT message missing parameters.")
        return
    try:
        x = int(parts[1])
        y = int(parts[2])
    except ValueError:
        logger.error("Invalid coordinates in ROBOT message.")
        return

    direction = parts[3].upper()
    if direction not in direction_map:
        logger.error(f"Invalid robot direction: {direction}")
        return

    robot_position = {"x": x, "y": y, "dir": direction_map[direction]}
    logger.info(f"Robot position updated: {robot_position}")


def handle_obstacle(parts, logger):
    global obstacles_list
    if len(parts) < 4:
        logger.error("OBSTACLE message missing parameters.")
        return
    try:
        x = int(parts[1])
        y = int(parts[2])
        obstacle_id = int(parts[3])
    except ValueError:
        logger.error("Invalid obstacle parameters.")
        return

    facing = parts[4].upper() if len(parts) > 4 else "UNKNOWN"
    direction = direction_map.get(facing, 4)

    obstacle = {"x": x, "y": y, "id": obstacle_id, "d": direction}
    obstacles_list.append(obstacle)
    logger.info(f"Obstacle recorded: {obstacle}")


def parse_obstacle_json(obstacle_data, logger):
    required_fields = ["x", "y", "id", "d"]
    if not all(field in obstacle_data for field in required_fields):
        logger.error("OBSTACLE JSON missing required fields.")
        return None
    d_val = obstacle_data["d"]
    if isinstance(d_val, str):
        d_val = direction_map.get(d_val.upper(), 4)
    return {
        "x": obstacle_data["x"],
        "y": obstacle_data["y"],
        "id": obstacle_data["id"],
        "d": d_val,
    }


def handle_target(parts, logger):
    if len(parts) < 3:
        logger.error("TARGET message missing parameters.")
        return
    obstacle_number = parts[1]
    target_id = parts[2]
    logger.info(f"Target identified: obstacle {obstacle_number}, id {target_id}")


def handle_status(parts, logger):
    if len(parts) < 2:
        logger.error("STATUS message missing parameters.")
        return
    status = parts[1]
    logger.info(f"Status update: {status}")


def handle_face(parts, logger):
    if len(parts) < 3:
        logger.error("FACE message missing parameters.")
        return
    obstacle_number = parts[1]
    side = parts[2]
    logger.info(f"Face update: obstacle {obstacle_number}, side {side}")


def handle_move(parts, logger):
    if len(parts) < 2:
        logger.error("MOVE message missing parameters.")
        return
    direction = parts[1]
    logger.info(f"Movement command: {direction}")


def send_obstacle_data(logger):
    if not obstacles_list:
        logger.warning("No obstacles recorded yet.")
        return

    map_data = {
        "obstacles": obstacles_list,
        "retrying": True,
        "robot_x": robot_position["x"],
        "robot_y": robot_position["y"],
        "robot_dir": robot_position["dir"],
        "big_turn": 0,
    }

    logger.info(f"Uploading map data to Algorithm Server: {map_data}")

    try:
        response = requests.post(ALGO_API_URL, json=map_data, timeout=5)
        if response.status_code == 200:
            logger.info("Successfully uploaded map data to Algorithm Server.")
            logger.info(f"Response: {response.json()}")
        else:
            logger.error(
                f"Failed to upload map data. Status: {response.status_code}, Response: {response.text}"
            )
    except requests.RequestException as e:
        logger.error(f"Error sending map data to Algorithm Server: {e}")


def handle_cmd(parts, logger, client_sock):
    global obstacles_list
    if len(parts) < 2:
        logger.error("CMD message missing parameters.")
        return

    command = parts[1].strip().lower()
    if command == "sendarena":
        if len(parts) < 3:
            logger.error("CMD,sendArena missing JSON data.")
            return
        try:
            arena_data = json.loads(parts[2])
            robot_position["x"] = arena_data.get("robot_x", robot_position["x"])
            robot_position["y"] = arena_data.get("robot_y", robot_position["y"])

            parsed_obstacles = []
            for obs in arena_data.get("obstacles", []):
                parsed = parse_obstacle_json(obs, logger)
                if parsed is not None:
                    parsed_obstacles.append(parsed)
            obstacles_list = parsed_obstacles

            send_obstacle_data(logger)

        except json.JSONDecodeError:
            logger.error("Parsing error for sendArena JSON data.")
    elif command == "beginexplore":
        logger.info("Task switched to Exploration Mode")
        send_obstacle_data(logger)
        send_text_message(client_sock, "Task switched to Exploration Mode", logger)
    elif command == "beginfastest":
        logger.info("Task switched to Fastest Path Mode")
        send_text_message(client_sock, "Task switched to Fastest Path Mode", logger)
    else:
        logger.warning(f"Unknown CMD command: {command}")


def send_text_message(client_sock, message, logger):
    try:
        client_sock.send(message.encode("utf-8"))
        logger.info(f"Sent text: {message}")
    except Exception as e:
        logger.error(f"Error sending text: {e}")


if __name__ == "__main__":
    start_bluetooth_service()
