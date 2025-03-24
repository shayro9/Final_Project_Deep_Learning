import torch
import os
from tqdm import tqdm


class SelfSupervisedTrainer:
    def __init__(self, model, train_loader, val_loader, loss_fn, optimizer, device, scheduler=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

    def train(self, num_epochs, checkpoints):
        if os.path.isfile(checkpoints):
            os.remove(checkpoints)
        self.model.train()
        for epoch in range(num_epochs):
            total_train_loss = 0
            for images, _ in self.train_loader:
                images = images.to(self.device)

                self.optimizer.zero_grad()
                reconstructed = self.model(images)
                loss = self.loss_fn(reconstructed, images)
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(self.train_loader)

            avg_val_loss = self.validate()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            if self.scheduler:
                self.scheduler.step()

        torch.save(self.model.encoder.state_dict(), checkpoints)

    def validate(self):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for images, _ in self.val_loader:
                images = images.to(self.device)
                reconstructed = self.model(images)
                loss = self.loss_fn(reconstructed, images)
                total_val_loss += loss.item()

        return total_val_loss / len(self.val_loader)


class ClassifierTrainer:
    def __init__(self, classifier, encoder, train_loader, test_loader, fn_loss, optimizer, device, scheduler=None):
        self.classifier = classifier.to(device)
        self.encoder = encoder.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.fn_loss = fn_loss
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

        self.encoder.eval()

    def train(self, num_epochs, checkpoints):
        best_val_accuracy = 0.0

        if os.path.isfile(checkpoints):
            os.remove(checkpoints)

        for epoch in range(num_epochs):
            self.classifier.train()
            total_loss = 0
            correct_predictions = 0
            total_samples = 0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                with torch.no_grad():
                    features = self.encoder(images)

                self.optimizer.zero_grad()
                outputs = self.classifier(features)
                loss = self.fn_loss(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            accuracy = (correct_predictions / total_samples) * 100
            val_accuracy = self.validate()
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(self.train_loader):.4f}, Accuracy: {accuracy:.2f}, Validation Accuracy: {val_accuracy:.2f} %")
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(self.classifier.state_dict(), checkpoints)
            if self.scheduler:
                self.scheduler.step()

    def validate(self):
        self.classifier.eval()
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                features = self.encoder(images)
                outputs = self.classifier(features)
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = (correct_predictions / total_samples) * 100

        return accuracy


class ClassificationGuidedTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, fn_loss, optimizer, device, scheduler=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.fn_loss = fn_loss
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

    def train(self, num_epochs, checkpoints):
        if os.path.isfile(checkpoints):
            os.remove(checkpoints)
        best_val_accuracy = 0.0
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            correct_predictions = 0
            total_samples = 0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.fn_loss(outputs, labels)
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                total_loss += loss.item()

            accuracy = (correct_predictions / total_samples) * 100
            val_accuracy = self.validate()
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(self.train_loader):.4f}, Accuracy: {accuracy:.2f}, Validation Accuracy: {val_accuracy:.2f}%")
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(self.model.state_dict(), checkpoints)
            if self.scheduler:
                self.scheduler.step()

    def validate(self):
        self.model.eval()
        total_val_loss = 0
        correct_val_predictions = 0
        total_val_samples = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.fn_loss(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                correct_val_predictions += (predicted == labels).sum().item()
                total_val_samples += labels.size(0)

                total_val_loss += loss.item()

        val_loss = total_val_loss / len(self.val_loader)
        val_accuracy = (correct_val_predictions / total_val_samples) * 100
        return val_accuracy


class simCLRTrainer:
    def __init__(self, epochs, model, optimizer, loss_fn, temperature=0.5, device='cuda'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.temperature = temperature
        self.device = device
        self.loss_fn = loss_fn
        self.epochs = epochs

    def train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0.0
        bar = tqdm(train_loader, desc="Training", leave=False)
        for (x1, x2, _) in bar:
            x1, x2 = x1.to(self.device), x2.to(self.device)
            x = torch.cat([x1, x2], dim=0)

            self.optimizer.zero_grad()
            z = self.model(x)
            loss = self.loss_fn(z, self.temperature)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * x1.size(0)

        avg_loss = epoch_loss / len(train_loader.dataset)
        return avg_loss

    def validate_epoch(self, val_loader):
        with torch.no_grad():
            self.model.eval()
            epoch_loss = 0.0
            bar = tqdm(val_loader, desc="Validating", leave=False)
            for (x1, x2, _) in bar:
                x1, x2 = x1.to(self.device), x2.to(self.device)
                x = torch.cat([x1, x2], dim=0)

                z = self.model(x)
                loss = self.loss_fn(z, self.temperature)
                epoch_loss += loss.item() * x1.size(0)

            avg_loss = epoch_loss / len(val_loader.dataset)
            return avg_loss

    def train_classifier(self, train_loader, optimizer, loss_fn):
        self.model.encoder.eval()
        self.model.classifier_head.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            bar = tqdm(train_loader, desc="Training classifier", leave=False)
            for x, y in bar:
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()

                with torch.no_grad():
                    features = self.model.encoder(x)

                logits = self.model.classifier_head(features)

                loss = loss_fn(logits, y)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            avg_loss = epoch_loss / len(train_loader.dataset)
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    def test_epoch(self, test_loader):
        with torch.no_grad():
            self.model.eval()
            epoch_correct = 0.0
            epoch_total = 0.0
            bar = tqdm(test_loader, desc="Testing", leave=False)
            for (x, y) in bar:
                x = x.to(self.device)
                y = y.to(self.device)

                z = self.model.encoder(x)
                logits = self.model.classifier_head(z)
                pred = logits.argmax(dim=1)
                epoch_correct += (pred == y).sum().item()
                epoch_total += y.size(0)

            accuracy = epoch_correct / epoch_total * 100
            return accuracy

    def fit(self, train_loader, val_loader):
        for epoch in range(self.epochs):
            print(f"------Epoch [{epoch + 1}/{self.epochs}]:--------")
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)
            print(f"Training Loss: {train_loss:.4f}  |  Validation Loss: {val_loss:.4f}")

    def test(self, test_loader):
        test_accuracy = self.test_epoch(test_loader)
        print(f"Test Accuracy = {test_accuracy: .2f}%")
