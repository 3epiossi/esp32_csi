import os

import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


# from data_calibration import calibrate_amplitude_custom, calibrate_phase, calibrate_amplitude, dwn_noise, hampel

CLASS_FOLDER = os.path.join(os.path.dirname(__file__),"data")
SUBCARRIES_NUM = 38
WINDOWS_SIZE = 50
PHASE_MAX, PHASE_MIN =  3.1415, -3.1415 #3.1389
AMP_MIN, AMP_MAX = 0.0, 577.6582

class_to_idx = defaultdict(int)
class_to_idx["one"] = 1
class_to_idx["two"] = 2
def read_csi_from_csv(path_to_csv):
    """
    Read csi data(amplitude, phase) from .csv data

    :param path_to_csv: string
    :param is_five_hhz: boolean
    :param antenna_pairs: integer
    :return: (amplitudes, phases) => (np.array of shape(data len, num_of_subcarriers * antenna_pairs),
                                     np.array of shape(data len, num_of_subcarriers * antenna_pairs))
    """

    data = pd.read_csv(path_to_csv, header=None).values
    amplitudes = data[:,0::2]
    phases = data[:, 1::2]
    
    return amplitudes, phases


# def read_labels_from_csv(path_to_csv):
#     """
#     Read labels(human activities) from csv file

#     :param path_to_csv: string
#     :return: labels, np.array of shape(data_len, 1)
#     """

#     data = pd.read_csv(path_to_csv, header=None).values
#     labels = data[:, 1]

#     return labels


def read_all_csv_from_class(class_names):
    """
    Read csi and labels data from all folders in the dataset

    :return: amplitudes, phases, labels all of shape (data len, num of subcarriers)
    """
    
    final_amplitudes, final_phases, final_labels = np.empty((0, WINDOWS_SIZE, SUBCARRIES_NUM)), \
                                                   np.empty((0, WINDOWS_SIZE, SUBCARRIES_NUM)), \
                                                   np.empty((0))
                                                   
    print("Amplitudes shape:", final_amplitudes.shape)
    print("Phases shape:", final_phases.shape)
    for label, class_name in enumerate(class_names):
        class_dir_name = os.path.join(CLASS_FOLDER, class_name)
        for csv_name in os.listdir(class_dir_name):
            amplitudes, phases = read_csi_from_csv(os.path.join(class_dir_name, csv_name))
            # print("Amplitudes shape:", amplitudes.shape)
            # print("Phases shape:", phases.shape)
            if amplitudes.shape[0] < WINDOWS_SIZE or phases.shape[0] < WINDOWS_SIZE:
                continue
            amplitudes, phases = amplitudes[:WINDOWS_SIZE], phases[:WINDOWS_SIZE]  # fix the bug with the last element

            final_amplitudes = np.concatenate((final_amplitudes, np.array((amplitudes,))))
            final_phases = np.concatenate((final_phases, np.array((phases,))))
            final_labels = np.append(final_labels, label)
            print("Amplitudes shape:", final_amplitudes.shape)
            print("Phases shape:", final_phases.shape)

    return final_amplitudes, final_phases, final_labels



class CSIDataset(Dataset):
    def __init__(self, class_names, window_size=32, step=1):
        self.amplitudes, self.phases, self.labels = read_all_csv_from_class(class_names)
        # print("Amplitudes shape:", self.amplitudes.shape)
        # print("Phases shape:", self.phases.shape)
        # self.amplitudes = calibrate_amplitude(self.amplitudes)
        # self.phases = calibrate_phase(self.phases)

        self.amplitudes = np.clip(self.amplitudes, AMP_MIN, AMP_MAX)
        self.phases = np.clip(self.phases, PHASE_MIN, PHASE_MAX)

        self.window = window_size
        if window_size == -1:
            self.window = self.labels.shape[0] - 1

        self.step = step
    
    def __getitem__(self, index):
        return np.concatenate((self.amplitudes[index], self.phases[index]), axis=1), class_to_idx[self.labels[index]]
    def __len__(self):
        return self.labels.shape[0]


# class CSIDataset(Dataset):
#     """CSI Dataset."""

#     def __init__(self, csv_files, window_size=32, step=1):
#         from sklearn import decomposition

#         self.amplitudes, self.phases, self.labels = read_all_data_from_files(csv_files)

#         self.amplitudes = calibrate_amplitude(self.amplitudes)

#         pca = decomposition.PCA(n_components=3)

#         # self.phases[:, 0 * SUBCARRIES_NUM_FIVE_HHZ:1 * SUBCARRIES_NUM_FIVE_HHZ] = calibrate_amplitude(calibrate_phase(
#         #     self.phases[:, 0 * SUBCARRIES_NUM_FIVE_HHZ:1 * SUBCARRIES_NUM_FIVE_HHZ]))
#         # self.phases[:, 1 * SUBCARRIES_NUM_FIVE_HHZ:2 * SUBCARRIES_NUM_FIVE_HHZ] = calibrate_amplitude(calibrate_phase(
#         #     self.phases[:, 1 * SUBCARRIES_NUM_FIVE_HHZ:2 * SUBCARRIES_NUM_FIVE_HHZ]))
#         # self.phases[:, 2 * SUBCARRIES_NUM_FIVE_HHZ:3 * SUBCARRIES_NUM_FIVE_HHZ] = calibrate_amplitude(calibrate_phase(
#         #     self.phases[:, 2 * SUBCARRIES_NUM_FIVE_HHZ:3 * SUBCARRIES_NUM_FIVE_HHZ]))
#         # self.phases[:, 3 * SUBCARRIES_NUM_FIVE_HHZ:4 * SUBCARRIES_NUM_FIVE_HHZ] = calibrate_amplitude(calibrate_phase(
#         #     self.phases[:, 3 * SUBCARRIES_NUM_FIVE_HHZ:4 * SUBCARRIES_NUM_FIVE_HHZ]))

#         self.amplitudes_pca = []

#         data_len = self.phases.shape[0]
#         for i in range(self.phases.shape[1]):
#             # self.phases[:data_len, i] = dwn_noise(hampel(self.phases[:, i]))[:data_len]
#             self.amplitudes[:data_len, i] = dwn_noise(hampel(self.amplitudes[:, i]))[:data_len]

#         for i in range(4):pca.fit_transform(
#             self.amplitudes_pca.append(
#                 self.amplitudes[:, i * SUBCARRIES_NUM_FIVE_HHZ:(i + 1) * SUBCARRIES_NUM_FIVE_HHZ]))
#         self.amplitudes_pca = np.array(self.amplitudes_pca)
#         self.amplitudes_pca = self.amplitudes_pca.reshape((self.amplitudes_pca.shape[1], self.amplitudes_pca.shape[0] * self.amplitudes_pca.shape[2]))

#         self.label_keys = list(set(self.labels))
#         self.class_to_idx = {
#             "standing": 0,
#             "walking": 1,
#             "get_down": 2,
#             "sitting": 3,
#             "get_up": 4,
#             "lying": 5,
#             "no_person": 6
#         }
#         self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

#         self.window = window_size
#         if window_size == -1:
#             self.window = self.labels.shape[0] - 1

#         self.step = step

#     def __getitem__(self, idx):
#         if self.window == 0:
#             return np.append(self.amplitudes[idx], self.phases[idx]), self.class_to_idx[
#                 self.labels[idx + self.window - 1]]

#         idx = idx * self.step
#         all_xs, all_ys = [], []
#         # idx = idx * self.window

#         for index in range(idx, idx + self.window):
#             all_xs.append(np.append(self.amplitudes[index], self.amplitudes_pca[index]))
#             # all_ys.append(self.class_to_idx[self.labels[index]])

#         return np.array(all_xs), self.class_to_idx[self.labels[idx + self.window - 1]]
#         # return np.array(all_xs), np.array(all_ys)

#     def __len__(self):
#         # return self.labels.shape[0] // self.window
#         # return (self.labels.shape[0] - self.window
#         return int((self.labels.shape[0] - self.window) // self.step) + 1


if __name__ == '__main__':
    val_dataset = CSIDataset(class_to_idx.keys())

    dl = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=1)

    for i in dl:
        print(i[0].shape)
        print(i[1].shape)

        break

    print(val_dataset[0])
# if __name__ == '__main__':
#     # Example usage
#     class_names = os.listdir(CLASS_FOLDER)
#     print(class_names)
#     amplitudes, phases= read_all_csv_from_class(class_names)
#     print("Amplitudes shape:", amplitudes.shape)
#     print("Phases shape:", phases.shape)
#     print("Amplitudes:", amplitudes)
#     print("Phases:", phases)