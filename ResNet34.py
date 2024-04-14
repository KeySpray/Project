import torch
from torch import nn
from torchvision.models import resnet34, ResNet34_Weights
from tqdm import tqdm
from DataProcessing import train_loader_nn, test_loader_nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Initialize the Weight Transforms
weights = ResNet34_Weights.DEFAULT

# Initialize model and set weights
rn = resnet34(weights=weights)
rn.to(device)

# Modifying final layer to match our dataset
num_features = rn.fc.in_features
rn.fc = nn.Linear(num_features, 4).to(device)

# Defining loss and optimizer functions
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(rn.parameters(), lr=0.001)


num_epochs = 20  # Set the number of epochs

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

    # Validation step, if you have a validation dataset
    rn.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        correct = total = 0
        for images, labels in val_loop:
            images, labels = images.to(device), labels.to(device)
            outputs = rn(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f'Validation Accuracy: {100 * correct / total:.2f}%')

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

plot_training_loss(training_losses)
plot_accuracy(accuracies)