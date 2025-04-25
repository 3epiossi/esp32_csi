import torch
from torch.nn import functional as F
from tqdm import tqdm
import platform

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and platform.system() == "Darwin":
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_train_metric(model, dl, criterion, batch_size, msg, device=None):
    if device is None:
        device = get_device()

    model.eval()
    correct, total, total_loss = 0, 0, 0

    model.hidden = model.init_hidden(batch_size)

    for x_val, y_val in tqdm(dl, total=len(dl), desc=msg):
        if x_val.size(0) != batch_size:
            continue

        x_val = x_val.float().to(device)
        y_val = y_val.to(device)

        model.hidden = model.init_hidden(x_val.size(0))
        out = model(x_val)

        loss = criterion(out, y_val.long())
        total_loss += loss.item()

        preds = F.log_softmax(out, dim=1).argmax(dim=1)
        total += y_val.size(0)
        correct += (preds == y_val).sum().item()

    acc = correct / total if total > 0 else 0
    return total_loss, correct, total, acc
