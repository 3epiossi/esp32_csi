import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
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
    parser = argparse.ArgumentParser(description="Apply filter to raw data.")
    parser.add_argument('--path', type=str, help='raw data\'s path', required=True)
    args = parser.parse_args()
    amplitudes, phases = read_csi_from_csv(args.path)
    measurements = np.concatenate((amplitudes, phases), axis=1).astype(np.float32)

    # 轉換成 PyTorch Tensor
    torch_data = torch.from_numpy(measurements)

    # 執行濾波
    device = get_device()
    filtered_data = IIR_filter_torch(torch_data, alpha=0.1, device=device)

    num_dims = measurements.shape[1]

    fig, axes = plt.subplots(num_dims, 1, figsize=(10, 3 * num_dims), sharex=True)

    if num_dims == 1:
        axes = [axes]  # 保持 axes 可迭代

    for i, ax in enumerate(axes):
        ax.plot(measurements[:, i], label=f'Original (dim {i})', alpha=0.5)
        ax.plot(filtered_data[:, i], label=f'Filtered (dim {i})', linewidth=2)
        # ax.set_ylabel('Value')
        ax.set_title(f'Dimension {i}')
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel('Time step')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
