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
from tensorflow.keras.optimizers import legacy
import re

from dataprocess.dataset import create_csi_dataset, get_tf_dataset
# from models import *

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

def load_data(class_folder, batch_size, window_size, val_split=0.2, debug=False, logger=None):
    logging.info("Loading dataset from: %s", class_folder)
    class_names = [d for d in os.listdir(class_folder) if os.path.isdir(os.path.join(class_folder, d))]
    features, labels = create_csi_dataset(class_folder, class_names, window_size, debug=debug, logger=logger)
    input_dim = features[0].shape[1] if len(features) else 0
    logging.info(f"input_dim : {input_dim}")
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=val_split, random_state=42)
    train_ds = get_tf_dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_ds = get_tf_dataset(X_val, y_val, batch_size=batch_size, shuffle=False)
    logging.info("Training dataset size: %d samples", len(X_train))
    logging.info("Validation dataset size: %d samples", len(X_val))
    return train_ds, val_ds, class_names, input_dim

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

def build_onelstm_sequential(input_shape, hidden_dim, output_dim, dropout=0.2):
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.LSTM(hidden_dim, activation='tanh', return_sequences=False, unroll=True))
    model.add(layers.Dense(output_dim, activation='softmax'))
    return model



def train(args, logger):
    train_ds, val_ds, class_names, input_dim = load_data(args.data_path, args.batch_size, args.window_size, val_split=0.2, debug=args.debug, logger=logger)
    logger = logging.getLogger()

    model = build_onelstm_sequential(
        input_shape=(args.window_size, args.input_dim),
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim
    )
    model.summary(print_fn=logger.info)

    optimizer = legacy.Adam()
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # Create necessary directories
    saved_models_dir = os.path.join(args.output_dir, 'saved_models')
    os.makedirs(saved_models_dir, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(saved_models_dir, 'simple_lstm_best.keras'),
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

    # Load the best model and convert to TensorFlow Lite
    best_model_path = os.path.join(saved_models_dir, 'simple_lstm_best.keras')
    best_model = tf.keras.models.load_model(best_model_path)
    
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]  # <== 只允許 builtin
    converter._experimental_lower_tensor_list_ops = True  # <== 強制展開 TensorList（如適用）
    
    try:
        tflite_model = converter.convert()
        # tf.lite.experimental.Analyzer.analyze(model_content=tflite_model) 
        # Save the TensorFlow Lite model
        tflite_model_path = os.path.join(saved_models_dir, 'model.tflite')
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        
        logging.info(f"TensorFlow Lite model saved to: {tflite_model_path}")
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        def extract_op_name(node_name: str) -> str:
            # 取最後一段
            op_full = node_name.split("/")[-1]
            # 移除數字與底線結尾，例如：add_36 ➜ add
            op_base = re.sub(r'_\d+$', '', op_full)
            return op_base.lower()  # 全部轉小寫，對齊 TFLite Op 名稱
        print(set([extract_op_name(d['name']) for d in interpreter.get_tensor_details()]))
        
        # Convert to C header file
        model_h_path = os.path.join(args.output_dir, 'tfModel.h')
        
        try:
            import subprocess
            
            # Write header file beginning
            with open(model_h_path, 'w') as f:
                f.write(f'#define TF_NUM_INPUTS_TIMESTEP {args.window_size}\n')
                f.write(f'#define TF_NUM_INPUTS_SUBCARRIER {args.input_dim}\n')
                f.write(f'#define TF_NUM_INPUTS ({args.window_size}*{args.input_dim})\n')
                f.write(f'#define TF_NUM_OUTPUTS {args.output_dim}\n')
                f.write(f'#define TF_NUM_OPS {model.count_params()}\n')  # 或手動填寫 op 數
                f.write('\nconst unsigned char tfModel[] = {\n')
            
            # Convert tflite model to C array format
            with open(model_h_path, 'a') as f:
                result = subprocess.run(['xxd', '-i', tflite_model_path], 
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE, 
                                      text=True)
                
                if result.returncode == 0:
                    # Extract only the hex data (skip the variable declaration)
                    lines = result.stdout.split('\n')
                    hex_lines = [line for line in lines if line.strip() and not line.strip().startswith('unsigned char')]
                    f.write('\n'.join(hex_lines))
                else:
                    logging.error(f"xxd command failed: {result.stderr}")
                    return
            
            # Write header file ending
            with open(model_h_path, 'a') as f:
                f.write(f'\nunsigned int model_len = {len(tflite_model)};\n')
            
            logging.info(f"C header file saved to: {model_h_path}")
            
        except FileNotFoundError:
            logging.error("xxd command not found. Please install it or use an alternative method.")
            logging.info("You can manually convert the .tflite file to a C header using online tools or other methods.")
        except Exception as e:
            logging.error(f"Error creating C header file: {e}")
    
    except Exception as e:
        logging.error(f"TensorFlow Lite conversion failed: {e}")
        logging.info("You might need to simplify your model architecture for better TensorFlow Lite compatibility.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Simple LSTM CSI Classifier (Keras version)")
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.00146, help='Learning rate')
    parser.add_argument('--input-dim', type=int, default=13, help='Input feature dimension')
    parser.add_argument('--hidden-dim', type=int, default=16, help='Hidden layer dimension')
    parser.add_argument('--layers', type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('--output-dim', type=int, default=2, help='Number of output classes')
    parser.add_argument('--window-size', type=int, default=20, help='Size of window')
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
    logger = setup_logger(debug=args.debug, logfile_path=os.path.join(logfile_dir, args.logfile))
    train(args, logger)