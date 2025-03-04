#!/bin/bash
cd /home/mdp23/MDP_RPI
export PYTHONPATH=/home/mdp23/MDP_RPI
exec /home/mdp23/MDP_RPI/.venv/bin/python -m stm_comm.serial_comm
