import tensorflow as tf
import numpy as np
from tqdm import tqdm

def get_device():
    # Keras/TensorFlow 自動選擇可用設備
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        return '/GPU:0'
    else:
        return '/CPU:0'

def get_train_metric(model, dataset, loss_fn, batch_size, msg):
    """
    Args:
        model: Keras model
        dataset: tf.data.Dataset yielding (x, y)
        loss_fn: tf.keras.losses.Loss instance
        batch_size: int
        msg: str, tqdm description
    Returns:
        total_loss, correct, total, acc
    """
    total_loss = 0.0
    correct = 0
    total = 0

    # 設定為評估模式
    for x_val, y_val in tqdm(dataset, desc=msg):
        logits = model(x_val, training=False)
        loss = loss_fn(y_val, logits)
        total_loss += loss.numpy() * x_val.shape[0]

        preds = tf.argmax(logits, axis=1)
        labels = tf.cast(tf.reshape(y_val, [-1]), tf.int64)
        correct += tf.reduce_sum(tf.cast(preds == labels, tf.int32)).numpy()
        total += x_val.shape[0]

    acc = correct / total if total > 0 else 0
    return total_loss, correct, total, acc