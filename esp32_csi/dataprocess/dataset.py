import os
import argparse
import logging
import torch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dataprocess.filter import IIR_filter_torch
from dataprocess.read_csv import *
CLASS_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data")

class CSIDataset(Dataset):
    def __init__(self, class_names, window_size, debug=False):
        self.window_size = window_size
        self.amplitudes, self.phases, self.labels = read_all_csv_from_class(class_names, window_size, debug)
        self.amplitudes, self.phases = np.array(self.amplitudes), np.array(self.phases)
        self.amplitudes = (self.amplitudes - self.amplitudes.mean()) / (self.amplitudes.std() + 1e-8)
        self.features = [
            np.concatenate([amp, phase], axis=1)
            for amp, phase in zip(self.amplitudes, self.phases)
        ]
        # self.amplitudes = IIR_filter_torch(torch.from_numpy(self.amplitudes).float(), alpha=0.9)
        if debug:
            logger.debug(f"Dataset created with {len(self.labels)} samples")

    def __getitem__(self, index):
        feature = self.amplitudes[index]
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return feature, label

    def __len__(self):
        return len(self.labels)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CSI Data Loader with configurable subcarrier and sequence length")
    parser.add_argument('--window', type=int, default=84, help='Window length (default: 84)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--logfile', type=str, default='dataset.log', help='Log file name (default: dataset.log)')
    parser.add_argument('--output-dir', type=str, default=os.path.join(os.path.dirname(__file__), "..", "output"), help='Path to output folder')
    args = parser.parse_args()

    window_size = args.window
    debug = args.debug
    log_filename = args.logfile

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)

    logfile_dir = os.path.join(args.output_dir, 'logs')
    if not os.path.exists(logfile_dir):
        os.makedirs(logfile_dir)
    file_handler = logging.FileHandler(os.path.join(logfile_dir, args.logfile), mode='w')
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"window_size: {window_size}, Log file: {log_filename}")

    class_names = [d for d in os.listdir(CLASS_FOLDER) if os.path.isdir(os.path.join(CLASS_FOLDER, d))]
    dataset = CSIDataset(class_names, window_size, debug=debug)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for i, (x_batch, y_batch) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Loading dataset"):
        logger.info(f"Batch {i} - x_batch shape: {x_batch.shape}, y_batch: {y_batch.item()}")

