import bluetooth
import os
import sys
import json
import socket
import threading
import requests
from config.logging_config import loggers

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Web server endpoint for map data uploads
ALGO_API_URL = "http://localhost:8000/path"

# Global variables for robot state and obstacles list
obstacles_list = []
robot_position = {"x": 1, "y": 1, "dir": 0}
direction_map = {"NORTH": 0, "EAST": 2, "SOUTH": 4, "WEST": 6}

# Global variable to store the currently connected Bluetooth client socket
active_bt_client = None
STM_SOCKET_PATH = "/tmp/stm_ipc.sock"


def start_bluetooth_service():
    logger = loggers["bluetooth"]
    # Start the Bluetooth IPC listener in a separate thread
    bt_ipc_thread = threading.Thread(target=start_bt_ipc_listener, daemon=True)
    bt_ipc_thread.start()

    # Create a Bluetooth socket using RFCOMM
    server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    server_sock.bind(("", bluetooth.PORT_ANY))
    server_sock.listen(1)
    port = server_sock.getsockname()[1]
    logger.info(f"Bluetooth listening on RFCOMM port {port}")

    # Advertise the service so that clients can discover it
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
    global active_bt_client
    active_bt_client = client_sock
    while True:
        try:
            data = client_sock.recv(1024)
            if not data:
                logger.info("Client closed the connection.")
                break
            # Strip leading/trailing whitespace and split on commas,
            # filtering out empty parts.
            message_str = data.decode("utf-8").strip()
            parts = [p.strip() for p in message_str.split(",") if p.strip()]
            logger.info(f"Received raw message: '{message_str}'")
            logger.info(f"Parsed message parts: {parts}")
            process_message(parts, client_sock, logger)
        except Exception as e:
            logger.error(f"Error in client communication: {e}")
            break
    client_sock.close()
    active_bt_client = None


def process_message(parts, client_sock, logger):
    if not parts:
        logger.warning("Empty message parts after splitting.")
        return

    msg_type = parts[0].upper()
    # For CMD messages, if JSON data is included, ensure we preserve it.
    if msg_type == "CMD" and len(parts) < 3:
        # Re-split with a max of 3 parts if possible
        message_str = ",".join(parts)
        parts = [p.strip() for p in message_str.split(",") if p.strip()]

    if not parts:
        logger.warning("Empty message parts after re-splitting CMD message.")
        return

    # if msg_type == "ROBOT":
    #    handle_robot(parts, logger)
    if msg_type == "TARGET":
        handle_target(parts, logger)
    elif msg_type == "STATUS":
        handle_status(parts, logger)
    # elif msg_type == "OBSTACLE":
    #    handle_obstacle(parts, logger)
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
        logger.error(f"ROBOT message missing parameters: {parts}")
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
    logger.info(parts)
    if len(parts) < 4:
        logger.error(f"OBSTACLE message missing parameters: {parts}")
        return
    try:
        x = int(parts[1])
        y = int(parts[2])
        obstacle_id = int(parts[3])
    except ValueError:
        logger.error("Invalid obstacle parameters.")
        return

    # 'facing' is optional; default to "UNKNOWN" if not provided.
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
        logger.error(f"TARGET message missing parameters: {parts}")
        return
    obstacle_number = parts[1]
    target_id = parts[2]
    logger.info(f"Target identified: obstacle {obstacle_number}, id {target_id}")


def handle_status(parts, logger):
    if len(parts) < 2:
        logger.error(f"STATUS message missing parameters: {parts}")
        return
    status = parts[1]
    logger.info(f"Status update: {status}")


def handle_face(parts, logger):
    global obstacles_list
    if len(parts) < 3:
        logger.error(f"FACE message missing parameters: {parts}")
        return

    try:
        obstacle_number = int(parts[1])  # Obstacle ID
        new_facing = parts[2].upper()  # New direction
    except ValueError:
        logger.error("Invalid FACE parameters.")
        return

    if new_facing not in direction_map:
        logger.error(f"Invalid FACE direction: {new_facing}")
        return

    new_direction = direction_map[new_facing]

    # Find and update the obstacle if it exists
    for obstacle in obstacles_list:
        if obstacle["id"] == obstacle_number:
            logger.info(f"Updating obstacle {obstacle_number}: New face {new_facing}")
            # Remove the old obstacle entry
            obstacles_list.remove(obstacle)
            break

    # Add the updated obstacle with the new face orientation
    updated_obstacle = {
        "x": obstacle["x"],
        "y": obstacle["y"],
        "id": obstacle_number,
        "d": new_direction,
    }
    obstacles_list.append(updated_obstacle)

    logger.info(f"Obstacle updated: {updated_obstacle}")


def send_to_stm(letter: str, socket_path: str = STM_SOCKET_PATH):
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(socket_path)
        client.send((letter + "\n").encode("utf-8"))
    except Exception as e:
        client.close()


def handle_move(parts, logger):
    if len(parts) < 2:
        logger.error(f"MOVE message missing parameters: {parts}")
        return

    # Extract the direction from the MOVE command and normalize to lowercase.
    direction = parts[1].strip().lower()

    # Map the received direction to the corresponding STM command.
    command_map = {
        "f": "SF010",
        "r": "SB010",
        "fl": "LF090",
        "fr": "RF090",
        "bl": "LB090",
        "br": "RB090",
    }

    if direction not in command_map:
        logger.error(f"Invalid MOVE direction: {direction}")
        return

    command_to_send = command_map[direction]
    logger.info(f"Received MOVE command, sending command: {command_to_send} to STM")
    try:
        send_to_stm(command_to_send)
    except Exception as e:
        logger.error(f"movement command to STM error: {e}")


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
        response = requests.post(ALGO_API_URL, json=map_data, timeout=60)
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
    global obstacles_list, robot_position
    if len(parts) < 2:
        logger.error("CMD message missing parameters.")
        return

    command = parts[1].strip().lower()
    if command == "sendarena":
        if len(parts) < 3:
            logger.error("CMD,sendArena missing JSON data.")
            return
        try:
            json_data_str = ",".join(parts[2:])  # Join everything after 'sendArena'
            arena_data = json.loads(json_data_str)

            # Validate that required keys exist
            required_keys = ["robot_x", "robot_y", "robot_dir", "obstacles"]
            if not all(key in arena_data for key in required_keys):
                logger.error("Invalid arena data: Missing required fields.")
                return

            # Update robot position
            robot_position["x"] = arena_data["robot_x"]
            robot_position["y"] = arena_data["robot_y"]

            # Validate and update robot direction
            robot_dir_str = arena_data["robot_dir"].upper()
            if robot_dir_str in direction_map:
                robot_position["dir"] = direction_map[robot_dir_str]
            else:
                logger.warning(
                    f"Invalid robot_dir '{robot_dir_str}', keeping previous direction."
                )

            # Parse obstacles
            parsed_obstacles = []
            for obs in arena_data["obstacles"]:
                # Ensure all required fields exist in obstacle data
                if not all(k in obs for k in ["x", "y", "id", "d"]):
                    logger.warning(f"Skipping invalid obstacle data: {obs}")
                    continue  # Skip malformed obstacles

                # Convert direction from string to numerical value
                obs_direction = obs["d"].upper()
                if obs_direction in direction_map:
                    obs["d"] = direction_map[obs_direction]
                else:
                    logger.warning(
                        f"Invalid obstacle direction '{obs_direction}', defaulting to SOUTH (4)."
                    )
                    obs["d"] = 4  # Default SOUTH

                parsed_obstacles.append(obs)

            # Update the global obstacles list
            obstacles_list.clear()
            obstacles_list.extend(parsed_obstacles)
            logger.info(
                f"Arena data updated: Robot {robot_position}, Obstacles: {obstacles_list}"
            )
        except json.JSONDecodeError:
            logger.error("Parsing error for sendArena JSON data.")
    elif command == "beginexplore":
        logger.info("Task switched to Exploration Mode")
        send_obstacle_data(logger)
        send_text_message(client_sock, "Task switched to Exploration Mode", logger)
    elif command == "beginfastest":
        logger.info("Task switched to Fastest Path Mode")
        send_text_message(client_sock, "Task switched to Fastest Path Mode", logger)
    elif command == "resetmap":
        obstacles_list.clear()
        robot_position = {"x": 1, "y": 1, "dir": 0}
        logger.info("Map reset: obstacles cleared and robot position reset.")
        send_text_message(client_sock, "Map reset", logger)
    else:
        logger.warning(f"Unknown CMD command: {command}")


def send_text_message(client_sock, message, logger):
    try:
        client_sock.send(message.encode("utf-8"))
        logger.info(f"Sent text: {message}")
    except Exception as e:
        logger.error(f"Error sending text: {e}")


def start_bt_ipc_listener():
    """
    Starts a Unix Domain Socket listener for receiving notifications from the STM service.
    When a notification is received, it is forwarded to the currently connected Bluetooth client.
    """
    logger = loggers["bluetooth"]
    bt_socket_path = "/tmp/bt_ipc.sock"
    if os.path.exists(bt_socket_path):
        os.remove(bt_socket_path)
    ipc_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    ipc_sock.bind(bt_socket_path)
    ipc_sock.listen(5)
    logger.info("Bluetooth IPC listener started on " + bt_socket_path)

    while True:
        conn, _ = ipc_sock.accept()
        try:
            data = conn.recv(1024)
            if data:
                notification = data.decode("utf-8").strip()
                logger.info("Received notification via IPC: " + notification)
                # Forward the notification to the active Bluetooth client, if connected
                global active_bt_client
                if active_bt_client:
                    try:
                        active_bt_client.send(notification.encode("utf-8"))
                        logger.info("Forwarded notification to Bluetooth client.")
                    except Exception as e:
                        logger.error(f"Error sending notification to BT client: {e}")
                else:
                    logger.warning(
                        "No active Bluetooth client to forward notification."
                    )
        except Exception as e:
            logger.error("Error handling BT IPC connection: " + str(e))
        finally:
            conn.close()


if __name__ == "__main__":
    start_bluetooth_service()
