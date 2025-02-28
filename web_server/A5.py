import serial
import time

def send_command(command):
    try:
        ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)  # Change to correct port
        ser.flush()
        
        ser.write(command.encode())  # Send command
        print(f"Sent: {command}")
        
        time.sleep(0.1)  # Short delay for processing
        response = ser.readline().decode('utf-8').strip()
        print(f"Received: {response}")
        
        ser.close()
    except serial.SerialException as e:
        print(f"Serial error: {e}")

if __name__ == "__main__":
    while True:
        cmd = input("Enter movement command (or 'exit' to quit): ").strip().upper()
        if cmd == "EXIT":
            break
        if len(cmd) == 5:
            send_command(cmd)
        else:
            print("Invalid command. Please enter a 5-character command.")