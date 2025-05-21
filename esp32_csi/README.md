# Neatlab esp32 CSI(channel state information) project
![](https://github.com/3epiossi/Neatlab/blob/main/doge.png)
This project is still unfinished, and I'm just a college student who doesn’t know much yet. So if you have any questions about how this project works, please don’t open an issue—I probably don’t know the answer either 😅

## Acknowledgments
1. **Espressif Official**: 
   Without your [esp-idf](https://github.com/espressif/esp-idf) and [esp-csi](https://github.com/espressif/esp-csi) examples, I wouldn’t even know where to start.

2. **Retsediv**: Thank you for your [WIFI_CSI_based_HAR](https://github.com/Retsediv/WIFI_CSI_based_HAR) project—your detailed documentation matched my needs perfectly.

## DONE and TODO
- [x] TX sends CSI to RX
- [x] Read and parse CSI info
- [x] Label data based on terminal input
- [x] Wrap all data into a dataset usable by a DataLoader
- [x] Load dataset into LSTM model for training
- [x] Optimize the project's code to improve readability.
- [x] Use argparse and logger for debugging 
- [x] Visualize loss curve and confusion matrix
- [x] Collect a large amount of experimental CSI data.
  * It can be considered as completing a part of the project, which can distinguish whether there are plastic bottles inside boxes or bags.
- [ ] Send the trained model parameters to ESP32 with TinyML so that the ESP32 can perform prediction(edge computing). 
  * Starting from this task, I will use Keras to restructure the entire project for easier deployment of TinyML.
- [ ] Test the model in practice

## Get Started
#### Hardware
- Two ESP32-32U modules

### Virtual Environment
- Miniconda

### Python Version
- 3.10.16

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/3epiossi/Neatlab.git
   cd esp32_csi
   ```
2. Set Up Python Environment
   Activate your Python virtual environment. This project uses Python 3.10.16 (Miniconda recommended):
   ```bash
   conda activate <your_env_name>
   ```
3. Install Python Dependencies
   ```bash
   pip install -r requirements.txt
   ```
4. **Install ESP-IDF**
   Follow the [official ESP-IDF documentation](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/) to install the latest version of **ESP-IDF** for your platform (macOS/Windows/Linux).
5. Build and flash the sender firmware
   ```bash
   cd csi_send
   idf.py set-target esp32
   idf.py build
   idf.py -p <PORT> [-b BAUD] flash
   ```
6. Build and flash the receiver firmware
   ```bash
   cd csi_send
   idf.py set-target esp32
   idf.py build
   idf.py -p <PORT> [-b BAUD] flash
   ```
7. **Collect Data Points**
   
   Run the following command:
   ```bash
   python csi_data_read_parse.py -p <PORT> -c <label_class_name>
   ```
   This will open a window displaying real-time CSI data visualization.
   Once you close the window, the program will terminate and one data point will be collected and saved.
   ![](https://github.com/3epiossi/Neatlab/blob/main/esp32_csi/data_collect.png)
8. **Train the LSTM Model**
   ```bash
   cd ~/Neatlab/esp32_csi
   python train_lstm.py
   ```
   This program will train LSTM model base on your data collected in step 7, and gives you Loss curve and Confusion matrix as the result.
   ![](https://github.com/3epiossi/Neatlab/blob/main/esp32_csi/result.png)
## License
This project is licensed under the GNU License – see the [LICENSE.md](https://github.com/3epiossi/Neatlab/blob/main/esp32_csi/LICENSE.md) file for details