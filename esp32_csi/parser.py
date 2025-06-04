#!/usr/bin/env python3
# -*-coding:utf-8-*-

# Copyright 2021 Espressif Systems (Shanghai) PTE LTD
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# WARNING: we don't check for Python build-time dependencies until
# check_environment() function below. If possible, avoid importing
# any external libraries here - put in external script, or import in
# their specific function instead.

import sys
import csv
import json
import argparse
import pandas as pd
import numpy as np

import serial
import re
import os
from os import path
from io import StringIO

from PyQt5.Qt import *
from pyqtgraph import PlotWidget
from PyQt5 import QtCore
import pyqtgraph as pq

import threading
import time

from itertools import chain
from datetime import datetime


# Reduce displayed waveforms to avoid display freezes
LLTF = 1
HT_LFT = 1
LLTF_INTERVAL = (26//LLTF + 1)
HT_LFT_INTERVAL = (28//HT_LFT + 1)
INPUT_DIM = 1 + (LLTF << 1) + (HT_LFT << 1)
# Remove invalid subcarriers
# secondary channel : below, HT, 40 MHz, non STBC, v, HT-LFT: 0~63, -64~-1, 384
csi_vaid_subcarrier_color = []

csi_vaid_subcarrier_color += [(255, 255, 255)]
# LLTF: 52
csi_vaid_subcarrier_color += [(255, 0, 0) for i in range(1,  26, LLTF_INTERVAL)]
csi_vaid_subcarrier_color += [(0, 255, 0) for i in range(1,  26, LLTF_INTERVAL)]
# HT-LFT: 56 + 56
csi_vaid_subcarrier_color += [(0, 0, 255) for i in range(1,  28, HT_LFT_INTERVAL)]
csi_vaid_subcarrier_color += [(255, 255, 0) for i in range(1, 28 , HT_LFT_INTERVAL)]
# csi_vaid_subcarrier_index += [i for i in range(124, 162)]  # 28  White
# csi_vaid_subcarrier_index += [i for i in range(163, 191)]  # 28  White

WINDOW_SIZE = 200  # buffer size
csi_data_array = np.zeros(
    [WINDOW_SIZE, INPUT_DIM], dtype=np.float32)

class csi_data_graphical_amplitude_window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSI Amplitude Data")
        self.resize(1280, 720)
        self.plotWidget_ted = PlotWidget(self)
        self.plotWidget_ted.setGeometry(QtCore.QRect(0, 0, 1280, 720))

        self.plotWidget_ted.setYRange(0, 100)
        self.plotWidget_ted.addLegend()

        self.curve_list = []

        # print(f"csi_vaid_subcarrier_color, len: {len(csi_vaid_subcarrier_color)}, {csi_vaid_subcarrier_color}")

        for i in range(INPUT_DIM):
            curve = self.plotWidget_ted.plot(
                csi_data_array[:, i], name=str(i), pen=csi_vaid_subcarrier_color[i])
            self.curve_list.append(curve)

        self.timer = pq.QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(100)

    def update_data(self):
        for i in range(INPUT_DIM):
            self.curve_list[i].setData(csi_data_array[:, i])


def csi_data_read_parse(port: str, csv_writer, csv_file, log_file_fd):
    ser = serial.Serial(port=port, baudrate=921600,
                        bytesize=8, parity='N', stopbits=1)
    if ser.isOpen():
        print("open success")
    else:
        print("open failed")
        return

    while True:
        line = ser.readline()
        strings = line.decode(errors='ignore').strip()
        if not strings:
            print("not strings")
            continue
        # print(strings)
        data = []
        try:
            raw_data = [float(value) for value in strings.split()]
            data.extend(raw_data[0:1])
            for j in range(1, 27, LLTF_INTERVAL):
                data.extend(raw_data[j:j+1])
            for j in range(27, 53, LLTF_INTERVAL):
                data.extend(raw_data[j:j+1])
            for j in range(53, 81, HT_LFT_INTERVAL):
                data.extend(raw_data[j:j+1])
            for j in range(81, 109, HT_LFT_INTERVAL):
                data.extend(raw_data[j:j+1])
        except:
            print("transform to numbers fail")
            continue
        data = np.array(data)
        if len(data) != INPUT_DIM:
            print(f"element number is not equal: {len(data)}")
            log_file_fd.write(f"element number is not equal: {len(data)}\n")
            log_file_fd.write(strings + '\n')
            log_file_fd.flush()
            continue


        # Rotate data to the left
        csi_data_array[:-1] = csi_data_array[1:]
        csi_data_array[-1] = data.astype(np.float32)
        print(csi_data_array[-1])
        try:
            csv_writer.writerow(csi_data_array[-1].tolist())
            csv_file.flush()  # 若在 class 裡
        except Exception as e:
            print("寫入 CSV 發生錯誤：", e)
    ser.close()
    return


class SubThread (QThread):
    def __init__(self, serial_port, class_name, log_file_name):
        super().__init__()
        self.serial_port = serial_port
        
        class_dir_name = os.path.join(os.path.dirname(__file__),'data', f'{class_name}')
        os.makedirs(class_dir_name, exist_ok=True)
        new_file_name = f"{datetime.now()}.csv"
        absolute_file_path = os.path.join(class_dir_name, new_file_name)
        self.csv_file = open(absolute_file_path, 'a')
        self.csv_writer = csv.writer(self.csv_file)
        self.log_file_fd = open(log_file_name, 'w')

    def run(self):
        csi_data_read_parse(self.serial_port, self.csv_writer, self.csv_file, self.log_file_fd)

    def __del__(self):
        self.csv_file.close()
        # self.csv_writer.close()
        # self.wait()
        self.log_file_fd.close()
        print("Thread exit")


if __name__ == '__main__':
    if sys.version_info < (3, 6):
        print(" Python version should >= 3.6")
        exit()

    parser = argparse.ArgumentParser(
        description="Read CSI data from serial port and display it graphically")
    parser.add_argument('-p', '--port', dest='port', action='store', required=True,
                        help="Serial port number of csv_recv device")
    parser.add_argument('-c', '--classify', dest='class_name', action='store', required=True,
                        help="Classify the data and save it to a file")
    parser.add_argument('-l', '--log', dest="log_file", action="store", default=f"{os.path.dirname(__file__)}/output/logs/csi_data.log",
                        help="Save other serial data the bad CSI data to a log file")
    

    args = parser.parse_args()
    serial_port = args.port
    class_name = args.class_name
    log_file_name = args.log_file
    app = QApplication(sys.argv)

    subthread = SubThread(serial_port, class_name, log_file_name)
    subthread.start()

    amplitude_window = csi_data_graphical_amplitude_window()
    amplitude_window.show()

    exit_code = app.exec()

    subthread.terminate()
    subthread.wait()
    subthread.csv_file.close()
    subthread.log_file_fd.close()

    sys.exit(exit_code)
