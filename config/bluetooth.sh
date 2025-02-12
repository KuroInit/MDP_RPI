#!/bin/bash

#NOTE: systemd script for bluetooth
sudo systemctl start bluetooth

bluetoothctl <<EOF
power on
discoverable on
pairable on
agent on
default-agent
EOF

echo "Bluetooth is configured and discoverable."
