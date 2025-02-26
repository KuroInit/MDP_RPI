import os
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


def send_command(ser, command):
    try:
        ser.write((command + "\n").encode("utf-8"))
        logger.info(f"Sent command: {command}")
    except Exception as e:
        logger.error(f"Error sending command: {e}")


def send_path(ser, path):
    try:
        ser.write("START_PATH\n".encode("utf-8"))
        for waypoint in path:
            line = f"{waypoint[0]},{waypoint[1]}\n"
            ser.write(line.encode("utf-8"))
        ser.write("END_PATH\n".encode("utf-8"))
        logger.info(f"Sent path: {path}")
    except Exception as e:
        logger.error(f"Error sending path: {e}")


def read_response(ser):
    try:
        response = ser.readline().decode("utf-8").strip()
        logger.info(f"Received response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error reading response: {e}")
        return None


def start_ipc_server(ser, socket_path="/tmp/stm_ipc.sock"):
    # Remove the existing socket file if it exists.
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
                # Process the command by sending it over the serial interface
                send_command(ser, command)
                # Optionally, read and log a response from STM
                response = read_response(ser)
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
        # Start the IPC server to listen for commands from other services
        start_ipc_server(ser)
        ser.close()
