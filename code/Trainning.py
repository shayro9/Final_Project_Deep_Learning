import abc
import torch


class Trainer(abc.ABC):
    def __init__(self, model, loss_fn, optimizer, device="cpu"):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        model.to(self.device)

    def fit(self, dl_train, dl_valid, epochs=100, verbose=True, early_stopping=5):
        train_loss, train_acc, valid_loss, valid_acc = [], [], [], []
        best_acc = None
        epochs_without_improvement = 0
        for epoch in range(epochs):
            if verbose:
                print(f"--- EPOCH {epoch + 1}/{epochs} ---")
            t_losses, t_accuracy = self.train_epoch(dl_train, verbose=verbose)
            train_loss += t_losses
            train_acc += [t_accuracy]

            v_losses, v_accuracy = self.test_epoch(dl_valid, verbose=verbose)
            valid_loss += v_losses
            valid_acc += [v_accuracy]

            if verbose:
                train_avg_loss = sum(t_losses) / len(t_losses)
                valid_avg_loss = sum(v_losses) / len(v_losses)
                print(f"Train: {train_avg_loss} , {t_accuracy} , Valid: {valid_avg_loss} , {v_accuracy}")

            if best_acc is None or v_accuracy > best_acc:
                best_acc = v_accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if early_stopping and epochs_without_improvement > early_stopping:
                    break

        return train_loss, train_acc, valid_loss, valid_acc

    def train_epoch(self, dl_train, **kw):
        self.model.train(True)
        losses = []
        num_samples = len(dl_train.sampler)
        num_correct = 0
        for batch in dl_train:
            loss, correct = self.train_batch(batch)
            losses.append(loss)
            num_correct += correct

        accuracy = 100 * num_correct / num_samples
        return losses, accuracy

    @abc.abstractmethod
    def train_batch(self, batch):
        raise NotImplementedError

    def test_epoch(self, dl_valid, **kw):
        self.model.train(False)
        losses = []
        num_samples = len(dl_valid.sampler)
        num_correct = 0
        for batch in dl_valid:
            loss, correct = self.test_batch(batch)
            losses.append(loss)
            num_correct += correct

        accuracy = 100 * num_correct / num_samples
        return losses, accuracy

    @abc.abstractmethod
    def test_batch(self, batch):
        raise NotImplementedError


class SelfSupervisedTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer, device=None):
        super().__init__(model, loss_fn, optimizer, device)

    def fit(self, dl_train, dl_valid, epochs=100, verbose=True, early_stopping=5):
        super().fit(dl_train, dl_valid, epochs, verbose, early_stopping)

    def train_batch(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        classify = self.model.to_classify

        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.loss_fn(output, y if classify else x)
        loss.backward()
        self.optimizer.step()

        num_correct = 0
        if classify:
            y_pred = torch.argmax(output, dim=1)
            num_correct = (y == y_pred).sum()

        return loss.item(), num_correct

    def test_batch(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        classify = self.model.to_classify

        with torch.no_grad():
            output = self.model(x)
            loss = self.loss_fn(output, y if classify else x)

            num_correct = 0
            if classify:
                y_pred = torch.argmax(output, dim=1)
                num_correct = (y == y_pred).sum()

        return loss.item(), num_correct
