import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import argparse
import logging
import os

class SimpleLSTMClassifier(keras.Model):
    def __init__(self, input_dim, hidden_dim, num_layers_lstm, out_classes_num, logger, dropout=0.0, bidirectional=False):
        super().__init__()
        self.logger = logger
        self.hidden_dim = hidden_dim
        self.layer_dim = num_layers_lstm
        self.bidirectional = bidirectional

        lstm_layers = []
        for i in range(num_layers_lstm):
            return_sequences = (i < num_layers_lstm - 1)
            lstm = layers.LSTM(
                hidden_dim,
                return_sequences=return_sequences,
                dropout=dropout,
                recurrent_dropout=0.0,
                name=f'lstm_{i+1}',
            )
            if bidirectional:
                lstm = layers.Bidirectional(lstm, name=f'bidir_lstm_{i+1}')
            lstm_layers.append(lstm)
        self.lstm_stack = keras.Sequential(lstm_layers)
        self.hidden2label = layers.Dense(out_classes_num, activation=None)

        logger.info("Initialized Keras LSTM model")
        logger.debug(f"Input dim: {input_dim}, Hidden dim: {hidden_dim}, "
                     f"Layers: {num_layers_lstm}, Bidirectional: {bidirectional}, "
                     f"Output classes: {out_classes_num}")

    def call(self, x, training=False):
        self.logger.debug(f"Forward input shape: {x.shape}")
        out = self.lstm_stack(x, training=training)
        self.logger.debug(f"LSTM output shape: {out.shape}")
        out = self.hidden2label(out)
        self.logger.debug(f"Output logits shape: {out.shape}")
        return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple LSTM Classifier (Keras version) with logging")
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--logfile', type=str, default='lstm_model.log', help='Log file name (default: lstm_model.log)')
    parser.add_argument('--output-dir', type=str, default=os.path.join(os.path.dirname(__file__), "..", "output"), help='Path to output folder')
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)

    logfile_dir = os.path.join(args.output_dir, 'logs')
    if not os.path.exists(logfile_dir):
        os.makedirs(logfile_dir)
    file_handler = logging.FileHandler(os.path.join(logfile_dir, args.logfile), mode='w')
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Example usage
    input_dim = 76
    hidden_dim = 128
    num_layers_lstm = 2
    out_classes_num = 5
    bidirectional = True
    dropout = 0.2

    model = SimpleLSTMClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers_lstm=num_layers_lstm,
        out_classes_num=out_classes_num,
        logger=logger,
        dropout=dropout,
        bidirectional=bidirectional
    )

    dummy_input = tf.random.normal((4, 84, input_dim))
    output = model(dummy_input, training=False)
    logger.info(f"Model output shape: {output.shape}")