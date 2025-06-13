# Neatlab esp32 CSI(channel state information) project
![](/demo/doge.png)
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
- [x] Send the trained model parameters to ESP32 with TinyML so that the ESP32 can perform prediction(edge computing). 
  * Starting from this task, I will use Keras to restructure the entire project for easier deployment of TinyML.
- [x] Test the model in practice and optimize it.
  * Big problem: the system always predicts something inside the box—even when there’s nothing there.
  ![](/demo/something_inside.png)
  solved(not fully) : ![](/demo/confusion_matrix.png)

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

# 專案解釋
專案目的：
讓esp32可以自主發送帶有CSI的訊號，而且自主預測箱子裡面物品的材質。
以下是這個專案的結構
```bash
.
├── .gitignore
├── csi_recv_lstm
│   ├── CMakeLists.txt
│   ├── dependencies.lock
│   ├── main
│   │   ├── app_main.cpp
│   │   ├── best_model
│   │   │   └── tfModel.h
│   │   ├── CMakeLists.txt
│   │   ├── idf_component.yml
│   │   └── tfModel.h
│   ├── README.md
│   ├── run.sh
│   ├── sdkconfig
│   ├── sdkconfig.defaults
│   └── sdkconfig.old
├── csi_send
│   ├── CMakeLists.txt
│   ├── dependencies.lock
│   ├── main
│   │   ├── app_main.c
│   │   ├── CMakeLists.txt
│   │   └── idf_component.yml
│   ├── README.md
│   ├── sdkconfig
│   ├── sdkconfig.defaults
│   └── sdkconfig.old
├── dataprocess
│   ├── dataset.py
│   └── read_csv.py
├── LICENSE.md
├── parser.py
├── README.md
├── requirements.txt
└── train_lstm.py
```
以下我會挑出幾個重要檔案來解釋他們的功用：
1. csi_send:
   * esp32 csi發送端所需的程式碼
2. csi_recv_lstm:
   * esp32 csi接收端的程式碼，負責兩個工作（非同時負責）
      1. 當PC在收集資料時，csi_recv_lstm會回傳接受到的資料(recv模式)
      2. 使用PC訓練好的模型來預測時，csi_recv_lstm負責實際執行預測，並把預測結果回傳到PC(lstm模式)
      3. 兩者差異可以用以下圖片看出
      ![](/demo/csi_recv_lstm.png)
3. parser.py:
   * 解析csi_recv_lstm回傳的資料(recv模式)
   * 將解析好的資料放在作為標籤的資料夾
4. dataprocess:
   * 將已經做好標籤的資料夾的資料做解析與轉換
   * 輸出dataset，供train_lstm.py訓練使用
5. train_lstm.py：
   * 將dataset拿來訓練
   * 訓練好的模型會放在output資料夾中（程式運行中會自動生成output資料夾）

專案邏輯：
![](/demo/project_logic.png)

執行結果：
1. ![loss curve](/demo/confusion_matrix.png)
2. ![](/demo/loss_curve.png)
3. [示範影片](https://www.youoube.com/shorts/uEpaOsHhDUo)
