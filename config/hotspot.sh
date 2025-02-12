#!/bin/bash

#NOTE:bash script runs and changes network option from wifi to AP on exec

nmcli connection down "Ash Iphone"
nmcli connection up "MDP23"
