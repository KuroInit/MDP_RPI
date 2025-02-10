import multiprocessing
import threading
import time
import os
from loguru import logger

# Import all subsystems
from wifi_server.wifi_server import run_wifi_server
# from bluetooth_module.bluetooth_module import bluetooth_listener
# from stm_serial.stm_serial import serial_communication
# from image_recognition.image_recognition import process_camera_feed
# from motion_control.motion_control import execute_motion_commands
# from navigation_system.navigation_system import navigation_logic

# Logging Setup
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger.add(
    os.path.join(LOG_DIR, "system.log"),
    rotation="5MB",
    retention="10 days",
    level="INFO",
)
logger.add(
    os.path.join(LOG_DIR, "errors.log"),
    rotation="5MB",
    retention="10 days",
    level="ERROR",
)

# Process and Thread Storage
processes = {}
threads = {}


def start_processes():
    """Start multiprocessing tasks."""
    processes["WiFi_Server"] = multiprocessing.Process(
        target=run_wifi_server, daemon=True
    )
    processes["Image_Processing"] = multiprocessing.Process(
        target=process_camera_feed, daemon=True
    )
    processes["Navigation"] = multiprocessing.Process(
        target=navigation_logic, daemon=True
    )

    for name, process in processes.items():
        process.start()
        logger.info(f"[STARTED] {name} (PID: {process.pid})")


def start_threads():
    """Start multithreading tasks."""
    threads["Bluetooth"] = threading.Thread(target=bluetooth_listener, daemon=True)
    threads["STM_Serial"] = threading.Thread(target=serial_communication, daemon=True)
    threads["Motion_Control"] = threading.Thread(
        target=execute_motion_commands, daemon=True
    )

    for name, thread in threads.items():
        thread.start()
        logger.info(f"[STARTED] {name}")


def monitor_processes():
    """Monitor and restart crashed processes."""
    while True:
        for name, process in processes.items():
            if not process.is_alive():
                logger.error(f"[CRITICAL] {name} Crashed! Restarting...")
                processes[name] = multiprocessing.Process(
                    target=process._target, daemon=True
                )
                processes[name].start()
                logger.info(f"[RESTARTED] {name} (PID: {process.pid})")
        time.sleep(5)


def shutdown_all():
    """Terminate all processes and threads."""
    logger.info("[SHUTTING DOWN] Stopping all processes and threads...")
    for name, process in processes.items():
        process.terminate()
        logger.info(f"[TERMINATED] {name}")
    logger.info("[CLEAN EXIT] All processes stopped.")
    os._exit(0)


if __name__ == "__main__":
    logger.info("[LAUNCHING SYSTEM] Initializing...")

    try:
        start_processes()
        start_threads()
        monitor_processes()
    except KeyboardInterrupt:
        shutdown_all()
