import os
import sys

# Ensure the project root is in sys.path so that "config" can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import socket
import serial
import time
from config.logging_config import loggers

logger = loggers["stm_comm"]


def init_serial(port="/dev/ttyUSB0", baudrate=115200, timeout=1):
    while True:
        try:
            ser = serial.Serial(port, baudrate, timeout=timeout)
            logger.info("Serial port initialized.")
            return ser
        except Exception as e:
            logger.error(f"Error initializing serial port: {e}")
            logger.info("Retrying in 10 seconds...")
            time.sleep(10)


def notify_bluetooth(command: str, choice: int):
    bt_socket_path = "/tmp/bt_ipc.sock"
    if choice == '1':
        notification = f"NOTIFY_CMD:{command}"
    else:
        notification = f"TARGET,{command}"

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.connect(bt_socket_path)
        client.send((notification + "\n").encode("utf-8"))
        response = client.recv(1024)
        logger.info(f"Bluetooth service response: {response.decode('utf-8')}")
    except Exception as e:
        logger.error(f"Error notifying Bluetooth service: {e}")
    finally:
        client.close()


def send_command(ser, command):
    try:
        ser.write((command).encode())
        logger.info(f"SERIAL: Sent command: {command}")
    except Exception as e:
        logger.error(f"Error sending command: {e}")


def send_path(ser, path):
    try:
        ser.write("START_PATH\n".encode("utf-8"))
        for waypoint in path:
            ser.write(f"{waypoint[0]},{waypoint[1]}\n".encode("utf-8"))
        ser.write("END_PATH\n".encode("utf-8"))
        logger.info(f"Sent path: {path}")
    except Exception as e:
        logger.error(f"Error sending path: {e}")


def read_response(ser):
    try:
        response = ser.readline().decode("utf-8").strip()
        logger.info(f"SERIAL: Received response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error reading response: {e}")
        return None


def start_ipc_server(ser, socket_path="/tmp/stm_ipc.sock"):
    if os.path.exists(socket_path):
        os.remove(socket_path)
    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_sock.bind(socket_path)
    server_sock.listen(5)
    logger.info("STM IPC server listening on " + socket_path)

    while True:
        conn, _ = server_sock.accept()
        try:
            data = conn.recv(1024)
            if data:
                command = data.decode("utf-8").strip()
                logger.info("Received command via IPC: " + command)

                if command.upper() == "FIN":
                    logger.info("Received FIN command. Returning to wait state.")
                    conn.send(b"OK: FIN received, returning to wait state")
                    continue

                send_command(ser, command)

                ack_received = False
                response = ""
                while not ack_received:
                    response = read_response(ser)
                    if response and "A" in response:
                        ack_received = True
                        logger.info(f"ACK received from STM for command: {command}")
                    else:
                        logger.info("Waiting for ACK from STM...")
                        time.sleep(0.5)

                notify_bluetooth(command,1)
                conn.send(
                    ("OK: " + (response if response else "No response")).encode("utf-8")
                )

            else:
                logger.warning("No data received on IPC connection.")

        except Exception as e:
            logger.error("Error handling IPC connection: " + str(e))
        finally:
            conn.close()


if __name__ == "__main__":
    ser = init_serial()
    if ser:
        start_ipc_server(ser)
        ser.close()
