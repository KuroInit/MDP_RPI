import serial
import time

def send_command(command):
    try:
        ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)  # Change to correct port
        ser.flush()
        
        ser.write((command + '\n').encode())  # Send command
        print(f"Sent: {command}")
        
        time.sleep(0.1)  # Short delay for processing
        response = ser.readline().decode('utf-8').strip()
        print(f"Received: {response}")
        
        ser.close()
    except serial.SerialException as e:
        print(f"Serial error: {e}")

if __name__ == "__main__":
    commands = [
        "SF090",  # Move forward 90cm
        "SF060",  # Move forward 60cm
        "RB090",  # Backward right 90 degrees
        "SB050",  # Move backward 50cm
        "RB090"   # Backward right 90 degrees
    ]
    
    for cmd in commands:
        send_command(cmd)
        time.sleep(1)  # Wait before sending the next command