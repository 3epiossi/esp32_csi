import logging
import os
import argparse
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

from dataset import CSIDataset
from metrics import get_train_metric
from models import SimpleLSTMClassifier


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_logger(debug, logfile_path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File
    file_handler = logging.FileHandler(logfile_path, mode="w")
    file_handler.setFormatter(formatter)

    logger.handlers = []
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def load_data(class_folder, batch_size, window_size):
    logging.info("Loading dataset from: %s", class_folder)

    class_names = os.listdir(class_folder)
    train_dataset = CSIDataset(class_names, window_size)
    val_dataset = train_dataset  # 如果需要區分驗證資料，可以自行分割

    trn_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    logging.info("Dataset size: %d samples", len(train_dataset))
    return trn_dl, val_dl


def plot_epoch_loss(args, epoch_losses):
    """Plot the epoch-loss curve."""
    figure, axis = plt.subplots(1,2)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses[:][0], marker='o', label='Training Epoch Loss', color='blue', axis=axis[0])
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses[:][1], marker='o', label='Validation Epoch Loss', color='orange', axis=axis[0])
    plt.xlabel('Epoch', axis=axis[0])
    plt.ylabel('Loss', axis=axis[0])
    plt.title('Epoch-Loss Curve', axis=axis[0])
    plt.legend(axis=axis[0])
    plt.grid(axis=axis[0])
    figure_dir = os.path.join(args.output_dir, 'figures')
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    plt.savefig(os.path.join(figure_dir, 'loss_curve.png'))
    plt.show()
    return 


def plot_confusion_matrix(args, model, val_dl, device, class_names):
    """Plot the confusion matrix."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in val_dl:
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.to(device)
            outputs = model(x_batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    figure, axis = plt.subplots(1,2)
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names, ax=axis[1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    figure_dir = os.path.join(args.output_dir, 'figures')
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    plt.savefig(os.path.join(figure_dir, 'confusion_matrix.png'))
    plt.show()

def draw_results(args, model, val_dl, class_names, epoch_losses, device):
    """Draw the results."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in val_dl:
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.to(device)
            outputs = model(x_batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    figure, axis = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(args.output_dim))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=axis[1])
    axis[1].set_title("Confusion Matrix")

    # Loss Curve
    train_losses = [loss[0] for loss in epoch_losses]
    val_losses = [loss[1] for loss in epoch_losses]
    axis[0].plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Training Loss', color='blue')
    axis[0].plot(range(1, len(val_losses) + 1), val_losses, marker='o', label='Validation Loss', color='orange')
    axis[0].set_xlabel('Epoch')
    axis[0].set_ylabel('Loss')
    axis[0].set_title('Epoch-Loss Curve')
    axis[0].legend()
    axis[0].grid(True)

    figure.tight_layout()

    figure_dir = os.path.join(args.output_dir, 'figures')
    os.makedirs(figure_dir, exist_ok=True)
    plt.savefig(os.path.join(figure_dir, 'result.png'))
    plt.show()

    
def train(args):
    device = get_device()
    logging.info("Using device: %s", device)

    trn_dl, val_dl = load_data(args.data_path, args.batch_size, args.window_size)

    model = SimpleLSTMClassifier(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers_lstm=args.layers,
        out_classes_num=args.output_dim,
        device=device,
        logger=logging.getLogger(),
        dropout=args.dropout,
        bidirectional=args.bidirectional
    ).to(device).float()

    logging.info("Model initialized: %s", model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5)
    best_acc, trials = args.accuracy, 0

    epoch_losses = []  # Store epoch losses for plotting

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for i, (x_batch, y_batch) in tqdm(enumerate(trn_dl), total=len(trn_dl), desc=f"Epoch {epoch}"):
            if x_batch.size(0) != args.batch_size:
                continue

            x_batch, y_batch = x_batch.float().to(device), y_batch.to(device)
            out = model(x_batch)
            loss = criterion(out, y_batch.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()


        train_loss, train_correct, _, train_acc = get_train_metric(model, trn_dl, criterion, args.batch_size, "Training epoch: ")
        val_loss, val_correct, _, val_acc = get_train_metric(model, val_dl, criterion, args.batch_size, "Validation epoch: ")
        epoch_losses.append((train_loss / len(trn_dl), val_loss / len(val_dl)))


        logging.info(f'Epoch {epoch:03d} | '
                     f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%} | '
                     f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}')

        if val_acc > best_acc:
            trials = 0
            best_acc = val_acc
            os.makedirs('saved_models', exist_ok=True)
            torch.save(model.state_dict(), 'saved_models/simple_lstm_best.pt')
            logging.info("New best model saved with accuracy: %.2f%%", best_acc * 100)
        else:
            trials += 1
            if trials >= args.patience:
                logging.info("Early stopping at epoch %d", epoch)
                break

        scheduler.step(val_loss)

    # Plot epoch-loss curve
    # plot_epoch_loss(args,epoch_losses)

    # Plot confusion matrix
    class_names = os.listdir(args.data_path)
    draw_results(args, model, val_dl, class_names, epoch_losses, device)
    # plot_confusion_matrix(args, model, val_dl, device, class_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Simple LSTM CSI Classifier")
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.00146, help='Learning rate')
    parser.add_argument('--input-dim', type=int, default=76, help='Input feature dimension')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--output-dim', type=int, default=2, help='Number of output classes')
    parser.add_argument('--window-size', type=int, default=80, help='Size of window')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional LSTM')
    parser.add_argument('--logfile', type=str, default='train.log', help='Log file path')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--accuracy', type=float, default=0.9, help='accuracy threshold')
    parser.add_argument('--data-path', type=str, default=os.path.join(os.path.dirname(__file__), "data"), help='Path to data folder')
    parser.add_argument('--output-dir', type=str, default=os.path.join(os.path.dirname(__file__), "output"), help='Path to output folder')

    args = parser.parse_args()
    logfile_dir = os.path.join(args.output_dir, 'logs')
    if not os.path.exists(logfile_dir):
        os.makedirs(logfile_dir)
    setup_logger(debug=args.debug, logfile_path=os.path.join(logfile_dir, args.logfile))
    train(args)