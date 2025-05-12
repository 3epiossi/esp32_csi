import torch
import numpy as np
import matplotlib.pyplot as plt
from dataprocess.read_csv import read_csi_from_csv
def get_device():
    if torch.backends.mps.is_available():
        print("✅ MPS is available. Using MPS for acceleration.")
        return torch.device("mps")
    else:
        print("⚠️ MPS not available. Using CPU.")
        return torch.device("cpu")

def IIR_filter_torch(data: torch.Tensor, alpha: float = 0.9, device=None) -> torch.Tensor:
    """
    用 PyTorch 實作的一階 IIR 濾波器。
    x_t = alpha * z_t + (1 - alpha) * x_{t-1}
    """
    if device is None:
        device = get_device()
    
    data = data.to(device)
    filtered = torch.zeros_like(data, device=device)
    filtered[0] = data[0]
    
    for t in range(1, data.shape[0]):
        filtered[t] = alpha * data[t] + (1 - alpha) * filtered[t - 1]
    
    return filtered.cpu()

def kalman_filter_torch(data: torch.Tensor, Q=1e-5, R=1e-2, device=None) -> torch.Tensor:
    """
    對每一維的時間序列資料套用一維 Kalman 濾波器。
    
    data: (time_stamp, state_dim)
    Q: process noise covariance
    R: measurement noise covariance
    回傳: (time_stamp, state_dim) 濾波後資料
    """
    if device is None:
        device = get_device()
    data = data.to(device)

    T, D = data.shape
    filtered = torch.zeros_like(data, device=device)

    # 初始化：用第一筆資料當作初始估計值與不確定性
    x_est = data[0]  # 初始狀態估計
    P = torch.ones(D, device=device)  # 初始估計誤差

    for t in range(T):
        z = data[t]

        # Prediction Step
        x_pred = x_est
        P_pred = P + Q

        # Update Step
        K = P_pred / (P_pred + R)              # Kalman gain
        x_est = x_pred + K * (z - x_pred)      # 更新估計
        P = (1 - K) * P_pred                   # 更新估計誤差

        filtered[t] = x_est

    return filtered.cpu()

def main():
    # 模擬資料：100 個時間點，3 維狀態
    amplitudes, phases = read_csi_from_csv("/Users/chang/github/Neatlab/esp32_csi/data/metal/2025-05-09 14:56:49.929084.csv")
    measurements = np.concatenate((amplitudes, phases), axis=1).astype(np.float32)

    # 轉換成 PyTorch Tensor
    torch_data = torch.from_numpy(measurements)

    # 執行濾波
    device = get_device()
    filtered_data = IIR_filter_torch(torch_data, alpha=0.1, device=device)

    # 可視化第 0 維
    plt.figure(figsize=(10, 5))
    plt.plot(measurements[:, 0], label='Original (dim 0)', alpha=0.5)
    plt.plot(filtered_data[:, 0], label='Filtered (dim 0)', linewidth=2)
    plt.title('One-dimensional filtering example')
    plt.xlabel('Time step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
