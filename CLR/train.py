from tqdm import tqdm
import torch


def train_epoch(model, train_loader, optimizer, loss_fn, temperature, device, scheduler):
    model.train()
    epoch_loss, epoch_total = 0.0, 0
    for x1, x2, _ in train_loader:
        x1, x2 = x1.to(device), x2.to(device)
        out1 = model(x1)
        out2 = model(x2)

        loss = loss_fn(out1, out2, temperature, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_size = out1.size(0)
        epoch_total += batch_size
        epoch_loss += loss.item() * batch_size

    return epoch_loss / epoch_total


def train_classifier(model, train_loader, optimizer, loss_fn, device):
    model.train()
    epoch_correct, epoch_loss, epoch_total = 0.0, 0.0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        out = model.classify(x)

        loss = loss_fn(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = out.size(0)
        epoch_total += batch_size
        epoch_loss += loss.item() * batch_size
        pred = out.argmax(dim=1)
        epoch_correct += (pred == y).sum().item()

    acc = epoch_correct / epoch_total * 100
    avg_loss = epoch_loss / epoch_total
    return avg_loss, acc


def test_epoch(model, test_loader, loss_fn, device):
    with torch.no_grad():
        model.eval()
        epoch_correct, epoch_loss, epoch_total = 0.0, 0.0, 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model.classify(x)

            loss = loss_fn(out, y)

            batch_size = out.size(0)
            epoch_total += batch_size
            epoch_loss += loss.item() * batch_size
            pred = out.argmax(dim=1)
            epoch_correct += (pred == y).sum().item()

        acc = epoch_correct / epoch_total * 100
        avg_loss = epoch_loss / epoch_total
        return avg_loss, acc

