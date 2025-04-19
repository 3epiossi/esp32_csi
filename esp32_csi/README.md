# Neatlab esp32 CSI(channel state information) project
This project is still unfinished, and I'm just a college student who doesn’t know much yet. So if you have any questions about how this project works, please don’t open an issue—I probably don’t know the answer either 😅

### Acknowledgments
1. **Espressif Official**: 
   Without your [esp-idf](https://github.com/espressif/esp-idf) and [esp-csi](https://github.com/espressif/esp-csi) examples, I wouldn’t even know where to start.

2. **Retsediv**: Thank you for your [WIFI_CSI_based_HAR](https://github.com/Retsediv/WIFI_CSI_based_HAR) project—your detailed documentation matched my needs perfectly.

### DONE and TODO
- [x] TX sends CSI to RX
- [x] Read and parse CSI info
- [x] Label data based on terminal input
- [x] Wrap all data into a dataset usable by a DataLoader
- [ ] Load dataset into LSTM model for training
- [ ] Test the model in practice
- [ ] Use Kalman filter for denoising

### License
This project is licensed under the GNU License – see the LICENSE.md file for details