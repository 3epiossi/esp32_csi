import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

CLASS_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data")

def read_csi_from_csv(path_to_csv, debug=False, logger=None):
    try:
        data = pd.read_csv(path_to_csv, header=None).values
    except pd.errors.EmptyDataError:
        if debug and logger is not None:
            logger.info(f"Empty CSV file: {path_to_csv}")
        return np.empty((0, 0))


    if debug and logger is not None:
        logger.debug(f"Loaded CSV: {path_to_csv}")
        logger.debug(f"Shape - data: {data.shape}")

    return data

def read_all_csv_from_class(class_folder, class_names, window_size, debug=False, logger=None):
    final_amplitudes, final_labels = [], []

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(class_folder, class_name)
        if not os.path.isdir(class_dir):
            if debug and logger is not None:
                logger.debug(f"Skipping non-directory: {class_dir}")
            continue

        csv_files = [f for f in os.listdir(class_dir) if f.endswith('.csv')]
        if debug and logger is not None:
            logger.debug(f"Reading class '{class_name}' with label {label} ({len(csv_files)} files)")

        for csv_name in csv_files:
            csv_path = os.path.join(class_dir, csv_name)
            amplitudes = read_csi_from_csv(csv_path, debug, logger)

            if amplitudes.shape[0] < window_size:
                if debug and logger is not None:
                    logger.debug(f"Skipping short sequence: {csv_name}")
                continue
            i = 0
            rssi_value = 100.0
            while i < amplitudes.shape[0]-window_size:
                if amplitudes[i][0] - rssi_value >= 4:
                    break
                rssi_value = amplitudes[i][0]
                i += 1
            
            logger.debug(amplitudes[i:i+window_size])
            final_amplitudes.append(amplitudes[i:i+window_size])
            final_labels.append(label)

    if debug and logger is not None:
        logger.debug(f"Total samples loaded: {len(final_labels)}")

    return final_amplitudes, final_labels