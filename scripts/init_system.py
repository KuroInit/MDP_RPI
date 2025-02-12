#!/usr/bin/env python3
import subprocess
import os


def run_shell_script(script_path):
    if os.path.exists(script_path):
        print(f"Running script: {script_path}")
        subprocess.run(["bash", script_path], check=True)
    else:
        print(f"Script not found: {script_path}")


def enable_and_start_service(service_name):
    print(f"Enabling service: {service_name}")
    subprocess.run(["sudo", "systemctl", "enable", service_name], check=True)
    print(f"Starting service: {service_name}")
    subprocess.run(["sudo", "systemctl", "start", service_name], check=True)


def initialize_system():
    run_shell_script("config/hotspot_setup.sh")
    run_shell_script("config/bluetooth_setup.sh")

    subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)

    services = [
        "bluetooth.service",
        "webserver.service",
        "stm_comm.service",
    ]

    for service in services:
        enable_and_start_service(service)


if __name__ == "__main__":
    initialize_system()
    print("System initialization complete.")
