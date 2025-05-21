import os
import argparse
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataprocess.dataset import create_csi_dataset, get_tf_dataset
from models.SimpleLSTMClassifier import SimpleLSTMClassifier

def setup_logger(debug, logfile_path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(logfile_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

def load_data(class_folder, batch_size, window_size, val_split=0.2, debug=False):
    logging.info("Loading dataset from: %s", class_folder)
    class_names = [d for d in os.listdir(class_folder) if os.path.isdir(os.path.join(class_folder, d))]
    features, labels = create_csi_dataset(class_names, window_size, debug=debug)
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=val_split, random_state=42)
    train_ds = get_tf_dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_ds = get_tf_dataset(X_val, y_val, batch_size=batch_size, shuffle=False)
    logging.info("Training dataset size: %d samples", len(X_train))
    logging.info("Validation dataset size: %d samples", len(X_val))
    return train_ds, val_ds, class_names

def plot_results(history, model, val_ds, class_names, args):
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch-Loss Curve')
    plt.legend()
    figure_dir = os.path.join(args.output_dir, 'figures')
    os.makedirs(figure_dir, exist_ok=True)
    plt.savefig(os.path.join(figure_dir, 'loss_curve.png'))
    plt.close()

    # Confusion Matrix
    y_true, y_pred = [], []
    for x_batch, y_batch in val_ds:
        logits = model(x_batch, training=False)
        preds = tf.argmax(logits, axis=1).numpy()
        y_true.extend(y_batch.numpy().flatten())
        y_pred.extend(preds.flatten())
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(figure_dir, 'confusion_matrix.png'))
    plt.close()

def train(args):
    train_ds, val_ds, class_names = load_data(args.data_path, args.batch_size, args.window_size, val_split=0.2, debug=args.debug)
    logger = logging.getLogger()

    model = SimpleLSTMClassifier(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers_lstm=args.layers,
        out_classes_num=args.output_dim,
        logger=logger,
        dropout=args.dropout,
        bidirectional=args.bidirectional
    )

    model.build(input_shape=(None, args.window_size, args.input_dim))
    model.summary(print_fn=logger.info)

    optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.output_dir, 'saved_models', 'simple_lstm_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    plot_results(history, model, val_ds, class_names, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Simple LSTM CSI Classifier (Keras version)")
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.00146, help='Learning rate')
    parser.add_argument('--input-dim', type=int, default=38, help='Input feature dimension')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--output-dim', type=int, default=2, help='Number of output classes')
    parser.add_argument('--window-size', type=int, default=80, help='Size of window')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional LSTM')
    parser.add_argument('--logfile', type=str, default='train.log', help='Log file path')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--data-path', type=str, default=os.path.join(os.path.dirname(__file__), "data"), help='Path to data folder')
    parser.add_argument('--output-dir', type=str, default=os.path.join(os.path.dirname(__file__), "output"), help='Path to output folder')

    args = parser.parse_args()
    logfile_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(logfile_dir, exist_ok=True)
    setup_logger(debug=args.debug, logfile_path=os.path.join(logfile_dir, args.logfile))
    train(args)