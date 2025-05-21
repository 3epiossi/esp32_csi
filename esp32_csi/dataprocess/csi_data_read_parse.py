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
CSI_VAID_SUBCARRIER_INTERVAL = 3

# Remove invalid subcarriers
# secondary channel : below, HT, 40 MHz, non STBC, v, HT-LFT: 0~63, -64~-1, 384
csi_vaid_subcarrier_index = []
csi_vaid_subcarrier_color = []
color_step = 255 // (28 // CSI_VAID_SUBCARRIER_INTERVAL + 1)

# LLTF: 52
csi_vaid_subcarrier_index += [i for i in range(6, 32, CSI_VAID_SUBCARRIER_INTERVAL)]     # 26  red
csi_vaid_subcarrier_color += [(i * color_step, 0, 0) for i in range(1,  26 // CSI_VAID_SUBCARRIER_INTERVAL + 2)]
csi_vaid_subcarrier_index += [i for i in range(33, 59, CSI_VAID_SUBCARRIER_INTERVAL)]    # 26  green
csi_vaid_subcarrier_color += [(0, i * color_step, 0) for i in range(1,  26 // CSI_VAID_SUBCARRIER_INTERVAL + 2)]
CSI_DATA_LLFT_COLUMNS = len(csi_vaid_subcarrier_index)
# HT-LFT: 56 + 56
csi_vaid_subcarrier_index += [i for i in range(66, 94, CSI_VAID_SUBCARRIER_INTERVAL)]    # 28  blue
csi_vaid_subcarrier_color += [(0, 0, i * color_step) for i in range(1,  28 // CSI_VAID_SUBCARRIER_INTERVAL + 2)]
csi_vaid_subcarrier_index += [i for i in range(95, 123, CSI_VAID_SUBCARRIER_INTERVAL)]   # 28  White
csi_vaid_subcarrier_color += [(i * color_step, i * color_step, i * color_step) for i in range(1,  28 // CSI_VAID_SUBCARRIER_INTERVAL + 2)]
# csi_vaid_subcarrier_index += [i for i in range(124, 162)]  # 28  White
# csi_vaid_subcarrier_index += [i for i in range(163, 191)]  # 28  White

CSI_DATA_INDEX = 200  # buffer size
CSI_DATA_COLUMNS = len(csi_vaid_subcarrier_index)
DATA_COLUMNS_NAMES = ["type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding",
                      "sgi", "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", "ant", "sig_len", "rx_state", "len", "first_word", "data"]
csi_data_array = np.zeros(
    [CSI_DATA_INDEX, CSI_DATA_COLUMNS], dtype=np.complex64)

class csi_data_graphical_amplitude_window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSI Amplitude Data")
        self.resize(1280, 720)
        self.plotWidget_ted = PlotWidget(self)
        self.plotWidget_ted.setGeometry(QtCore.QRect(0, 0, 1280, 720))

        self.plotWidget_ted.setYRange(-20, 100)
        self.plotWidget_ted.addLegend()

        self.csi_amplitude_array = np.abs(csi_data_array)
        self.curve_list = []

        # print(f"csi_vaid_subcarrier_color, len: {len(csi_vaid_subcarrier_color)}, {csi_vaid_subcarrier_color}")

        for i in range(CSI_DATA_COLUMNS):
            curve = self.plotWidget_ted.plot(
                self.csi_amplitude_array[:, i], name=str(i), pen=csi_vaid_subcarrier_color[i])
            self.curve_list.append(curve)

        self.timer = pq.QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(100)

    def update_data(self):
        self.csi_amplitude_array = np.abs(csi_data_array)
        for i in range(CSI_DATA_COLUMNS):
            self.curve_list[i].setData(self.csi_amplitude_array[:, i])

# NOTE: The following class is similar to csi_data_graphical_amplitude_window
# but it displays the phase of the CSI data instead of the amplitude.
class csi_data_graphical_phase_window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSI Phase Data")
        self.resize(1280, 720)
        self.plotWidget_ted = PlotWidget(self)
        self.plotWidget_ted.setGeometry(QtCore.QRect(0, 0, 1280, 720))

        self.plotWidget_ted.setYRange(-np.pi, np.pi)
        self.plotWidget_ted.addLegend()

        self.csi_amplitude_array = np.angle(csi_data_array)
        self.curve_list = []

        # print(f"csi_vaid_subcarrier_color, len: {len(csi_vaid_subcarrier_color)}, {csi_vaid_subcarrier_color}")

        for i in range(CSI_DATA_COLUMNS):
            curve = self.plotWidget_ted.plot(
                self.csi_amplitude_array[:, i], name=str(i), pen=csi_vaid_subcarrier_color[i])
            self.curve_list.append(curve)

        self.timer = pq.QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(100)

    def update_data(self):
        self.csi_amplitude_array = np.angle(csi_data_array)
        for i in range(CSI_DATA_COLUMNS):
            self.curve_list[i].setData(self.csi_amplitude_array[:, i])


# NOTE: The following class is similar to csi_data_graphical_amplitude_window
# but it displays the I-Q graph of the CSI data instead of the amplitude.
class csi_data_graphical_polar_window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSI Polar Plot (IQ Data)")
        self.resize(1280, 720)

        self.plotWidget_ted = pq.GraphicsLayoutWidget(self)
        self.plotWidget_ted.setGeometry(QtCore.QRect(0, 0, 1280, 720))
        self.plot = self.plotWidget_ted.addPlot()
        self.plot.setAspectLocked(True)

        R = 50
        radius_step = 10
        angle_step = 30
        self.plot.setXRange(-R, R)
        self.plot.setYRange(-R, R)

        self.add_polar_grid(R, radius_step, angle_step)

        self.csi_real_array = np.real(csi_data_array[-1])
        self.csi_imag_array = np.imag(csi_data_array[-1])
        self.curve_list = []

        for i in range(CSI_DATA_COLUMNS):
            curve = self.plot.plot(
                [self.csi_real_array[i]], [self.csi_imag_array[i]],
                pen=None, symbol='o', symbolSize=9,
                symbolBrush=pq.mkBrush(csi_vaid_subcarrier_color[i])
            )
            self.curve_list.append(curve)

        self.timer = pq.QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(100)  

    def add_polar_grid(self, R, radius_step, angle_step):

        for angle in range(0, 360, angle_step):
            rad = np.radians(angle)
            x = R * np.cos(rad)
            y = R * np.sin(rad)
            self.plot.plot([0, x], [0, y], pen=pq.mkPen(color=(200, 200, 200), width=1))

            label_x = (R + 10) * np.cos(rad)
            label_y = (R + 10) * np.sin(rad)
            text = pq.TextItem(f"{angle}°", anchor=(0.5, 0.5))
            text.setPos(label_x, label_y)
            self.plot.addItem(text)
        
        for r in range(radius_step, R + 1, radius_step):
            circle = pq.QtWidgets.QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
            circle.setPen(pq.mkPen(color=(200, 200, 200), width=1))
            self.plot.addItem(circle)

    def update_data(self):
        self.csi_real_array = np.real(csi_data_array[-1])
        self.csi_imag_array = np.imag(csi_data_array[-1])

        for i in range(CSI_DATA_COLUMNS):
            self.curve_list[i].setData([self.csi_real_array[i]], [self.csi_imag_array[i]])


# NOTE: The following class is similar to csi_data_graphical_amplitude_window
# but it displays the amplitude, phase, and I-Q graph of the CSI data instead of the amplitude.
class csi_data_graphical_combined_window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSI Amplitude, Phase, and I-Q graph")
        self.showMaximized()

        layout = QGridLayout(self)

        self.amplitude_plot = PlotWidget(self)
        self.amplitude_plot.setYRange(-20, 100)
        self.amplitude_plot.addLegend()
        layout.addWidget(self.amplitude_plot, 0, 0)

        self.phase_plot = PlotWidget(self)
        self.phase_plot.setYRange(-np.pi, np.pi)
        self.phase_plot.addLegend()
        layout.addWidget(self.phase_plot, 1, 0)

        self.polar_plot_widget = pq.GraphicsLayoutWidget(self)
        self.polar_plot = self.polar_plot_widget.addPlot()
        self.polar_plot.setAspectLocked(True)  
        layout.addWidget(self.polar_plot_widget, 0, 1, 2, 1)  

        R = 50
        radius_step = 10
        angle_step = 30
        self.polar_plot.setXRange(-R, R)
        self.polar_plot.setYRange(-R, R)

        self.add_polar_grid(R, radius_step, angle_step)

        self.csi_amplitude_array = np.abs(csi_data_array)
        self.csi_phase_array = np.angle(csi_data_array)
        self.csi_real_array = np.real(csi_data_array[-1])
        self.csi_imag_array = np.imag(csi_data_array[-1])

        self.amplitude_curves = []
        self.phase_curves = []
        self.polar_curves = []

        for i in range(CSI_DATA_COLUMNS):
            amplitude_curve = self.amplitude_plot.plot(
                self.csi_amplitude_array[:, i], pen=csi_vaid_subcarrier_color[i]
            )
            self.amplitude_curves.append(amplitude_curve)

            phase_curve = self.phase_plot.plot(
                self.csi_phase_array[:, i], pen=csi_vaid_subcarrier_color[i]
            )
            self.phase_curves.append(phase_curve)

            polar_curve = self.polar_plot.plot(
                [self.csi_real_array[i]], [self.csi_imag_array[i]],
                pen=None, symbol='o', symbolSize=9,
                symbolBrush=pq.mkBrush(csi_vaid_subcarrier_color[i])
            )
            self.polar_curves.append(polar_curve)

        self.timer = pq.QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(100)  

    def add_polar_grid(self, R, radius_step, angle_step):

        for angle in range(0, 360, angle_step):
            rad = np.radians(angle)
            x = R * np.cos(rad)
            y = R * np.sin(rad)
            self.polar_plot.plot([0, x], [0, y], pen=pq.mkPen(color=(200, 200, 200), width=1))

            label_x = (R + 10) * np.cos(rad)
            label_y = (R + 10) * np.sin(rad)
            text = pq.TextItem(f"{angle}°", anchor=(0.5, 0.5))
            text.setPos(label_x, label_y)
            self.polar_plot.addItem(text)

        for r in range(radius_step, R + 1, radius_step):
            circle = pq.QtWidgets.QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
            circle.setPen(pq.mkPen(color=(200, 200, 200), width=1))
            self.polar_plot.addItem(circle)

    def update_data(self):

        self.csi_amplitude_array = np.abs(csi_data_array)
        self.csi_phase_array = np.angle(csi_data_array)
        self.csi_real_array = np.real(csi_data_array[-1])
        self.csi_imag_array = np.imag(csi_data_array[-1])

        for i in range(CSI_DATA_COLUMNS):
            self.amplitude_curves[i].setData(self.csi_amplitude_array[:, i])

        for i in range(CSI_DATA_COLUMNS):
            self.phase_curves[i].setData(self.csi_phase_array[:, i])

        for i in range(CSI_DATA_COLUMNS):
            self.polar_curves[i].setData([self.csi_real_array[i]], [self.csi_imag_array[i]])





def csi_data_read_parse(port: str, csv_writer, log_file_fd):
    ser = serial.Serial(port=port, baudrate=921600,
                        bytesize=8, parity='N', stopbits=1)
    if ser.isOpen():
        print("open success")
    else:
        print("open failed")
        return

    while True:
        strings = str(ser.readline())
        if not strings:
            break

        strings = strings.lstrip('b\'').rstrip('\\r\\n\'')
        index = strings.find('CSI_DATA')

        if index == -1:
            # Save serial output other than CSI data
            log_file_fd.write(strings + '\n')
            log_file_fd.flush()
            continue

        csv_reader = csv.reader(StringIO(strings))
        csi_data = next(csv_reader)

        if len(csi_data) != len(DATA_COLUMNS_NAMES):
            print("element number is not equal")
            log_file_fd.write("element number is not equal\n")
            log_file_fd.write(strings + '\n')
            log_file_fd.flush()
            continue

        try:
            csi_raw_data = json.loads(csi_data[-1])
        except json.JSONDecodeError:
            print("data is incomplete")
            log_file_fd.write("data is incomplete\n")
            log_file_fd.write(strings + '\n')
            log_file_fd.flush()
            continue

        # Reference on the length of CSI data and usable subcarriers
        # https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-guides/wifi.html#wi-fi-channel-state-information
        if len(csi_raw_data) != 128 and len(csi_raw_data) != 256 and len(csi_raw_data) != 384:
            print(f"element number is not equal: {len(csi_raw_data)}")
            log_file_fd.write(f"element number is not equal: {len(csi_raw_data)}\n")
            log_file_fd.write(strings + '\n')
            log_file_fd.flush()
            continue

        # Rotate data to the left
        csi_data_array[:-1] = csi_data_array[1:]
        if len(csi_raw_data) == 128:
            csi_vaid_subcarrier_len = CSI_DATA_LLFT_COLUMNS
        else:
            csi_vaid_subcarrier_len = CSI_DATA_COLUMNS

        for i in range(csi_vaid_subcarrier_len):
            csi_data_array[-1][i] = complex(csi_raw_data[csi_vaid_subcarrier_index[i] * 2 + 1],
                                            csi_raw_data[csi_vaid_subcarrier_index[i] * 2])
        
        append_data = np.array(list(map(lambda x : (np.abs(x), np.angle(x)),csi_data_array[-1]))).flatten()
        csv_writer.writerow(append_data)
    ser.close()
    return


class SubThread (QThread):
    def __init__(self, serial_port, class_name, log_file_name):
        super().__init__()
        self.serial_port = serial_port
        
        class_dir_name = os.path.join(os.path.dirname(__file__), '..' ,'data', f'{class_name}')
        os.makedirs(class_dir_name, exist_ok=True)
        new_file_name = f"{datetime.now()}.csv"
        absolute_file_path = os.path.join(class_dir_name, new_file_name)
        self.csv_file = open(absolute_file_path, 'a')
        self.csv_writer = csv.writer(self.csv_file)
        self.log_file_fd = open(log_file_name, 'w')

    def run(self):
        csi_data_read_parse(self.serial_port, self.csv_writer, self.log_file_fd)

    def __del__(self):
        self.csv_file.close()
        # self.csv_writer.close()
        self.wait()
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
    parser.add_argument('-l', '--log', dest="log_file", action="store", default=f"{os.path.dirname(__file__)}/../output/logs/csi_data.log",
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

    # phase_window = csi_data_graphical_phase_window()
    # phase_window.show()
    
    # polar_window = csi_data_graphical_polar_window()
    # polar_window.show()

    # combined_window = csi_data_graphical_combined_window()
    # combined_window.show()

    sys.exit(app.exec())
