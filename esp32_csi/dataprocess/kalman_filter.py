import torch
import numpy as np
import matplotlib.pyplot as plt
from .read_csv import read_csi_from_csv

def get_device():
    if torch.backends.mps.is_available():
        print("✅ MPS is available. Using MPS for acceleration.")
        return torch.device("mps")
    else:
        print("⚠️ MPS not available. Using CPU.")
        return torch.device("cpu")

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
    # 讀取資料
    amplitudes, phases = read_csi_from_csv("/Users/chang/github/Neatlab/esp32_csi/data/metal/2025-05-09 14:56:49.929084.csv")
    measurements = np.concatenate((amplitudes, phases), axis=1).astype(np.float32)
    torch_data = torch.from_numpy(measurements).float()

    device = get_device()
    filtered_data = kalman_filter_torch(torch_data, Q=1e-5, R=1e-2, device=device)

    # 可視化第 0 維
    plt.figure(figsize=(10, 5))
    plt.plot(measurements[:, 2], label='Original (dim 0)', alpha=0.5)
    plt.plot(filtered_data[:, 2], label='Kalman Filtered (dim 0)', linewidth=2)
    plt.title('Kalman Filtering on CSI Data (dim 0)')
    plt.xlabel('Time step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
