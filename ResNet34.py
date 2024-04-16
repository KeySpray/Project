import torch
from torch import nn
from torchvision.models import resnet34, ResNet34_Weights
from tqdm import tqdm
from DataProcessing import train_loader_nn, test_loader_nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def create_resnet(use_default, weights, num_classes):
    if use_default == True:
        rn = resnet34(weights=weights)
        rn.to(device)
        num_features = rn.fc.in_features
        rn.fc = nn.Linear(num_features, num_classes).to(device)
    else:
        rn = resnet34()
        rn.to(device)
        num_features = rn.fc.in_features
        rn.fc = nn.Linear(num_features, num_classes).to(device)
        state_dict = torch.load(weights)
        rn.load_state_dict(state_dict)
    return rn

def train_and_validate(rn):

    # Defining loss and optimizer functions
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(rn.parameters(), lr=0.001)

    training_losses = []
    accuracies = []

    # Train and validate model
    for epoch in range(num_epochs):
        rn.train()  # Set the model to training mode
        running_loss = 0.0
        train_loop = tqdm(train_loader_nn, desc=f"Training Epoch {epoch + 1}/{num_epochs}")
        for images, labels in train_loop:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = rn(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader_nn.dataset)
        training_losses.append(epoch_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        val_loop = tqdm(test_loader_nn, desc=f"Validation Epoch {epoch + 1}/{num_epochs}")

        # Validation step
        rn.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            correct = total = 0
            all_probs = []
            for images, labels in val_loop:
                images, labels = images.to(device), labels.to(device)
                outputs = rn(images)
                _, predicted = torch.max(outputs.data, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            accuracies.append(accuracy)
            print(f'Validation Accuracy: {100 * correct / total:.2f}%')
        if epoch_loss < 0.005:
            #torch.save(rn.state_dict(), 'resnet34_weights.pth')
            break
    
    return training_losses, accuracies

def validate(rn, val_loop):
    rn.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        correct = total = 0
        all_probs = []
        all_labels = []
        for images, labels in val_loop:
            images, labels = images.to(device), labels.to(device)
            outputs = rn(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        accuracy = accuracy_score(np.concatenate(all_labels), np.concatenate(all_probs).argmax(axis=1))
        precision = precision_score(np.concatenate(all_labels), np.concatenate(all_probs).argmax(axis=1), average='weighted')
        recall = recall_score(np.concatenate(all_labels), np.concatenate(all_probs).argmax(axis=1), average='weighted')
        f1 = f1_score(np.concatenate(all_labels), np.concatenate(all_probs).argmax(axis=1), average='weighted')
        cm = confusion_matrix(np.concatenate(all_labels), np.concatenate(all_probs).argmax(axis=1))
        
        print(f'Validation Accuracy: {accuracy:.2f}%')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1 Score: {f1:.2f}')
        print('Confusion Matrix:')
        print(cm)

    return accuracy, precision, recall, f1, cm


def plot_training_loss(training_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig("Training_Loss_vs_Epoch.png")

def plot_accuracy(accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(accuracies, label='Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig("Accuracy_vs_Epoch.png")

def plot_precision_recall_curves(precision, recall, classes):
    for i in range(len(classes)):
        plt.figure(figsize=(8, 6))
        plt.plot(recall[i], precision[i], marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve (Class {classes[i]})')
        plt.grid(True)
        filename = f"Precision_Recall_Curve_Class_{classes[i]}.png"
        plt.savefig(filename)
        plt.close()  # Close the figure to release memory


num_epochs = 150  # Set the number of epochs

classes = {'Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented'}
num_classes = 4


rn = create_resnet(False, 'resnet34_weights.pth', num_classes)
val_loop = tqdm(test_loader_nn)
accuracy, precision, recall, f1, cm = validate(rn, val_loop)

# plot_training_loss(training_losses)
# plot_accuracy(accuracies)