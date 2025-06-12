import os
import argparse
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from dataprocess.read_csv import *

csi_mean = 0.0
csi_std = 1.0

def create_csi_dataset(class_folder, class_names, window_size, debug=False, logger=None, args=None):
    global csi_mean, csi_std
    amplitudes, labels = read_all_csv_from_class(class_folder, class_names, window_size, debug, logger, args)
    amplitudes = np.array(amplitudes)
    csi_mean, csi_std = amplitudes.mean(), amplitudes.std()
    amplitudes = (amplitudes - amplitudes.mean()) / (amplitudes.std() + 1e-8)
    labels = np.array(labels)
    if debug:
        print(f"Dataset created with {len(labels)} samples")
    return amplitudes, labels

def get_tf_dataset(features, labels, batch_size=1, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(labels))
    ds = ds.batch(batch_size)
    return ds

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CSI Data Loader with configurable subcarrier and sequence length")
    parser.add_argument('--window', type=int, default=20, help='Window length (default: 84)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--logfile', type=str, default='dataset.log', help='Log file name (default: dataset.log)')
    parser.add_argument('--output-dir', type=str, default=os.path.join(os.path.dirname(__file__), "..", "output"), help='Path to output folder')
    parser.add_argument('--class-folder', type=str, default=os.path.join(os.path.dirname(__file__), "..", "data"), help='Path to class folder')
    parser.add_argument('--lltf', type=int, default=1, help='Specified number of subcarrier in LLTF')
    parser.add_argument('--ht-lft', type=int, default=1, help='Specified number of subcarrier in HT_LFT')
    args = parser.parse_args()

    window_size = args.window
    debug = args.debug
    log_filename = args.logfile
    class_folder = args.class_folder

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
    logger.info(f"class_folder: {class_folder}")

    class_names = [d for d in os.listdir(class_folder) if os.path.isdir(os.path.join(class_folder, d))]
    features, labels = create_csi_dataset(class_folder, class_names, window_size, debug=debug, logger=logger, args=args)
    dataset = get_tf_dataset(features, labels, batch_size=1, shuffle=False)

    for i, (x_batch, y_batch) in tqdm(enumerate(dataset), total=len(features), desc="Loading dataset"):
        logger.info(f"Batch {i} - x_batch shape: {x_batch.shape}, y_batch: {y_batch.numpy()}")
        print(x_batch.numpy(), y_batch.numpy())
        if i >= 2:
            break