import serial
import time
from config.logging_config import loggers

logger = loggers["stm_comm"]


# TODO: work with stm team to fix convertion factor if needed.
def init_serial(port="/dev/ttyS0", baudrate=115200, timeout=1):
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


if __name__ == "__main__":
    ser = init_serial()
    if ser:
        send_command(ser, "MOVE,f")
        print("Response:", read_response(ser))
        ser.close()
