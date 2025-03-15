import torch
import os

class SelfSupervisedTrainer:
    def __init__(self, model, train_loader, val_loader, loss_fn, optimizer, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

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

            # Validate after each epoch
            avg_val_loss = self.validate()

            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        
        torch.save(self.model.encoder.state_dict(), checkpoints)
        #print(f"Encoder saved to {path}")
        print("Training completed.")

    def validate(self):
        self.model.eval()  # Set model to evaluation mode (no gradient updates)
        total_val_loss = 0

        with torch.no_grad():  # Disable gradient calculations
            for images, _ in self.val_loader:
                images = images.to(self.device)
                reconstructed = self.model(images)
                loss = self.loss_fn(reconstructed, images)
                total_val_loss += loss.item()

        return total_val_loss / len(self.val_loader)




class ClassifierTrainer:
    def __init__(self, classifier, encoder, train_loader, test_loader, fn_loss, optimizer, device):
        self.classifier = classifier.to(device)
        self.encoder = encoder.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.fn_loss = fn_loss
        self.optimizer = optimizer
        self.device = device

        # Freeze encoder parameters
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

                with torch.no_grad():  # Don't compute gradients for encoder
                    features = self.encoder(images)  # Extract features

                self.optimizer.zero_grad()
                outputs = self.classifier(features)
                loss = self.fn_loss(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)  # Get the class with highest probability
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            accuracy = (correct_predictions / total_samples) * 100
            val_accuracy = self.validate()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(self.train_loader):.4f}, Accuracy: {accuracy:.2f}, Validation Accuracy: {val_accuracy:.2f} %")
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(self.classifier.state_dict(), checkpoints)
            
       # torch.save(self.classifier.state_dict(), checkpoints)
        print("Training completed.")

    def validate(self):
        self.classifier.eval()
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                features = self.encoder(images)  # Extract features
                outputs = self.classifier(features)
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = (correct_predictions / total_samples) * 100
        #print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy


class ClassificationGuidedTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, fn_loss, optimizer, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.fn_loss = fn_loss
        self.optimizer = optimizer
        self.device = device
    
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
                outputs = self.model(images)  # Get output from the model
                loss = self.fn_loss(outputs, labels)  # Compute loss
                loss.backward()
                self.optimizer.step()
        
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)  # Get the class with highest probability
                correct_predictions += (predicted == labels).sum().item()  # Compare to true labels
                total_samples += labels.size(0)
        
                total_loss += loss.item()
        
            accuracy = (correct_predictions / total_samples) * 100  # Accuracy as a percentage
            val_accuracy = self.validate()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(self.train_loader):.4f}, Accuracy: {accuracy:.2f}, Validation Accuracy: {val_accuracy:.2f}%")
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(self.model.state_dict(), checkpoints)

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