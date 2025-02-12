import bluetooth
import json
import time
import os
from config.logging_config import loggers
import sys

#print("current path")
#for path in sys.path:
#    print(" ", path)

#absolute path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# server init
def start_bluetooth_service():
    logger = loggers["bluetooth"]
    server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    server_sock.bind(("", bluetooth.PORT_ANY))
    server_sock.listen(1)
    port = server_sock.getsockname()[1]
    logger.info(f"Bluetooth listening on RFCOMM port {port}")

    # discoverable
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

    # message procesing according to doc
    # TODO: tell jonathan retrive image and results


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
        handle_cmd(parts, logger)
    elif msg_type == "SEND_IMAGE":
        file_path = parts[1] if len(parts) > 1 else None
        if file_path:
            send_image_file(client_sock, file_path, logger)
        else:
            logger.error("SEND_IMAGE message missing file path.")
    elif msg_type == "SEND_STRING":
        if len(parts) > 1:
            send_text_message(client_sock, ",".join(parts[1:]), logger)
        else:
            logger.error("SEND_STRING message missing content.")
    else:
        logger.warning(f"Unknown message type: {msg_type}")


def handle_robot(parts, logger):
    if len(parts) < 4:
        logger.error("ROBOT message missing parameters.")
        return
    x, y, direction = parts[1], parts[2], parts[3]
    logger.info(f"Robot position: x={x}, y={y}, direction={direction}")


def handle_target(parts, logger):
    if len(parts) < 3:
        logger.error("TARGET message missing parameters.")
        return
    obstacle_number, target_id = parts[1], parts[2]
    logger.info(f"Target: obstacle {obstacle_number}, id {target_id}")


def handle_status(parts, logger):
    if len(parts) < 2:
        logger.error("STATUS message missing parameters.")
        return
    status = parts[1]
    logger.info(f"Status: {status}")


def handle_obstacle(parts, logger):
    if len(parts) < 4:
        logger.error("OBSTACLE message missing parameters.")
        return
    x, y, number = parts[1], parts[2], parts[3]
    facing = parts[4] if len(parts) > 4 else "UNKNOWN"
    logger.info(f"Obstacle: x={x}, y={y}, number={number}, facing={facing}")


def handle_face(parts, logger):
    if len(parts) < 3:
        logger.error("FACE message missing parameters.")
        return
    obstacle_number, side = parts[1], parts[2]
    logger.info(f"Face update: obstacle={obstacle_number}, side={side}")


def handle_move(parts, logger):
    if len(parts) < 2:
        logger.error("MOVE message missing parameters.")
        return
    direction = parts[1]
    logger.info(f"Movement command: {direction}")


def handle_cmd(parts, logger):
    if len(parts) < 2:
        logger.error("CMD message missing parameters.")
        return
    command = parts[1]
    logger.info(f"Special command: {command}")


def send_image_file(client_sock, file_path, logger):
    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        return
    try:
        file_size = os.path.getsize(file_path)
        header = json.dumps(
            {"filename": os.path.basename(file_path), "filesize": file_size}
        )
        client_sock.send(header.encode("utf-8"))
        time.sleep(0.2)
        with open(file_path, "rb") as f:
            chunk = f.read(1024)
            while chunk:
                client_sock.send(chunk)
                chunk = f.read(1024)
        logger.info(f"Sent image: {file_path} ({file_size} bytes)")
    except Exception as e:
        logger.error(f"Error sending image: {e}")


def send_text_message(client_sock, message, logger):
    try:
        client_sock.send(message.encode("utf-8"))
        logger.info(f"Sent text: {message}")
    except Exception as e:
        logger.error(f"Error sending text: {e}")


if __name__ == "__main__":
    start_bluetooth_service()
