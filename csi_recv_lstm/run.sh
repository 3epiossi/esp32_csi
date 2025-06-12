#!/bin/zsh
set -e
# 設定目標為 esp32
# idf.py set-target esp32
# 編譯專案
idf.py build
idf.py -p /dev/cu.usbserial-0001 flash
idf.py monitor -p /dev/cu.usbserial-0001

