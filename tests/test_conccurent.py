import os
import time
from multiprocessing import Process
from loguru import logger

TEST_LOG_DIR = "test_log"
if not os.path.exists(TEST_LOG_DIR):
    os.makedirs(TEST_LOG_DIR)

logger.add(
    os.path.join(TEST_LOG_DIR, "test_concurrent.log"),
    rotation="5MB",
    retention="7 days",
    level="ERROR",
)


def start_bluetooth():
    try:
        from bluetooth.bluetooth_listener import start_bluetooth_service

        start_bluetooth_service()
    except Exception as e:
        logger.error("Bluetooth service error: {}", e)


def start_webserver():
    try:
        import uvicorn

        uvicorn.run("webserver.app:app", host="0.0.0.0", port=8000, log_level="info")
    except Exception as e:
        logger.error("Webserver service error: {}", e)


def start_stm():
    try:
        from stm_comm.serial_comm import init_serial

        ser = init_serial()
        while True:
            time.sleep(1)
    except Exception as e:
        logger.error("STM service error: {}", e)


if __name__ == "__main__":
    processes = [
        Process(target=start_bluetooth),
        Process(target=start_webserver),
        Process(target=start_stm),
    ]

    for p in processes:
        p.start()

    time.sleep(10)

    for p in processes:
        p.terminate()
    for p in processes:
        p.join()

    logger.info("Concurrent services test completed.")
    print("Test completed. Check the test_log/test_concurrent.log for any errors.")
