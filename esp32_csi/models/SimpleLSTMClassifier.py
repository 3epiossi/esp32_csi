import torch
from torch import nn
import logging
import argparse
import os

def get_device(logger):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA (GPU)")
    elif torch.backends.mps.is_available() :
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


class SimpleLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers_lstm, out_classes_num, device, logger, dropout=0.0, bidirectional=False):
        super().__init__()
        self.device = device
        self.logger = logger
        self.hidden_dim = hidden_dim
        self.layer_dim = num_layers_lstm
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers_lstm,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
            bias=True
        )

        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.hidden2label = nn.Linear(output_dim, out_classes_num)
        self.hidden = None

        logger.info("Initialized LSTM model")
        logger.debug(f"Input dim: {input_dim}, Hidden dim: {hidden_dim}, "
                     f"Layers: {num_layers_lstm}, Bidirectional: {bidirectional}, "
                     f"Output classes: {out_classes_num}")

    def forward(self, x):
        self.logger.debug(f"Forward input shape: {x.shape}")

        if self.hidden is None:
            self.hidden = self.init_hidden(x.size(0))

        out, _ = self.lstm(x, self.hidden)
        self.logger.debug(f"LSTM output shape: {out.shape}")

        out = self.hidden2label(out[:, -1, :])
        self.logger.debug(f"Output logits shape: {out.shape}")

        return out

    def init_hidden(self, batch_size):
        h0 = nn.Parameter(nn.init.xavier_uniform_(
            torch.empty(self.layer_dim, batch_size, self.hidden_dim).float()
        ), requires_grad=True).to(self.device)

        c0 = nn.Parameter(nn.init.xavier_uniform_(
            torch.empty(self.layer_dim, batch_size, self.hidden_dim).float()
        ), requires_grad=True).to(self.device)

        self.logger.debug(f"Initialized hidden states with shape: {h0.shape}")
        return h0, c0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple LSTM Classifier with platform-adaptive device and logging")
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

    device = get_device(logger)

    model = SimpleLSTMClassifier(
        input_dim=76, hidden_dim=128,
        num_layers_lstm=2, out_classes_num=5,
        device=device, logger=logger
    ).to(device)

    dummy_input = torch.randn(4, 84, 76).to(device)  
    output = model(dummy_input)
    logger.info(f"Model output shape: {output.shape}")
