import torch
from torch import nn
from torchvision.models import resnet34, ResNet34_Weights
from tqdm import tqdm
from DataProcessing import train_loader_nn, test_loader_nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def create_resnet(use_default, weights, num_classes):
    if use_default == True:
        rn = resnet34(weights=ResNet34_Weights)
        rn.to(device)
        num_features = rn.fc.in_features
        rn.fc = nn.Linear(num_features, num_classes).to(device)
    else:
        rn = resnet34(weights=None)
        rn.to(device)
        num_features = rn.fc.in_features
        rn.fc = nn.Linear(num_features, num_classes).to(device)
        if weights != None:
            state_dict = torch.load(weights)
            rn.load_state_dict(state_dict)
    return rn

def train_and_validate(rn, num_epochs, weight_label):

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
            accuracy = correct / total
            accuracies.append(accuracy)
            print(f'Validation Accuracy: {100 * correct / total:.2f}%')
        if epoch_loss < 0.005:
            torch.save(rn.state_dict(), f'{weight_label}')
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
        
        print(f'Validation Accuracy: {accuracy:.3f}%')
        print(f'Precision: {precision:.3f}')
        print(f'Recall: {recall:.3f}')
        print(f'F1 Score: {f1:.3f}')
        print('Confusion Matrix:')
        print(cm)

    return accuracy, precision, recall, f1, cm


def plot_loss_and_accuracy(training_losses, accuracies, label):
    epochs = range(1, len(training_losses) + 1)

    fig, ax1 = plt.subplots()

    # Plot training loss
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(epochs, training_losses, label='Training Loss', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for accuracy
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(epochs, accuracies, label='Accuracy', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    min_value = min(min(training_losses), min(accuracies))  # Minimum value for both y-axes
    max_value = max(max(training_losses), max(accuracies))  # Maximum value for both y-axes
    ax1.set_ylim(min_value, max_value)  # Set y-axis limits for training loss axis
    ax2.set_ylim(min_value, max_value)  # Set y-axis limits for accuracy axis

    ax1.set_yticks([i/10 for i in range(int(min_value*10), int(max_value*10)+1)])  # Set y-ticks for training loss axis
    ax2.set_yticks([i/10 for i in range(int(min_value*10), int(max_value*10)+1)])  # Set y-ticks for accuracy axis

    plt.title(f'Training Loss and Accuracy vs. Epoch - {label}')
    plt.savefig(f'Training_Loss_Accuracy_vs_Epoch_{label}')


def plot_precision_recall_f1(rn, val_loader, label):
    all_preds = []
    all_labels = []
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = rn(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)

    fig, ax = plt.subplots()
    ax.bar(range(len(precision)), precision, label='Precision')
    ax.bar(range(len(recall)), recall, label='Recall')
    ax.bar(range(len(f1)), f1, label='F1-score')
    ax.set_xticks(range(len(precision)))
    ax.set_xticklabels(classes, rotation=45)
    ax.set_xlabel('Classes')
    ax.set_ylabel('Score')
    ax.set_title('Precision, Recall, and F1-score by Class - ' + label)
    ax.legend()
    plt.savefig(f'Precision_Recall_F1_{label}.png')





num_epochs = 150  # Set the number of epochs

classes = {'Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented'}
num_classes = 4

############


# Section of code to create pretrained resnet model, fine-tune, and validate with saved weights after fine-tuning


############

# Create pretrained ResNet model, fine-tune on our dataset
rn_pretrained = create_resnet(True, ResNet34_Weights, num_classes)
# Train and validate will save model weights
weight_label = 'resnet34_weights_pretrained.pth'
training_losses_pretrained, accuracies_pretrained = train_and_validate(rn_pretrained, num_epochs, weight_label)
plot_loss_and_accuracy(training_losses_pretrained, accuracies_pretrained, 'Pretrained')




############


# Section of code to create untrained resnet model, train, and validate with saved weights after training


############

# Create untrained ResNet model, train on our dataset
rn_untrained = create_resnet(False, None, num_classes)
# Train and validate will save model weights
weight_label = 'resnet34_weights_untrained.pth'
training_losses_untrained, accuracies_untrained = train_and_validate(rn_untrained, num_epochs, weight_label)
plot_loss_and_accuracy(training_losses_untrained, accuracies_untrained, 'Untrained')

# Load pretrained, fine-tuned ResNet model with saved weights
rn_pretrained = create_resnet(False, 'resnet34_weights_pretrained.pth', num_classes)
# Create validation tqdm loop with test dataset
val_loop = tqdm(test_loader_nn)
# Validate test dataset, printing and returning various metrics
accuracy_pretrained, precision_pretrained, recall_pretrained, f1_pretrained, cm_pretrained = validate(rn_pretrained, val_loop)
# Plot Precision, Recall, and F1-score by Class for pretrained model
plot_precision_recall_f1(rn_pretrained, val_loop, 'Pretrained')

# Load previously untrained ResNet model with saved weights
rn_pretrained = create_resnet(False, 'resnet34_weights_untrained.pth', num_classes)
# Create validation tqdm loop with test dataset
val_loop = tqdm(test_loader_nn)
# Validate test dataset, printing and returning various metrics
accuracy_untrained, precision_untrained, recall_untrained, f1_untrained, cm_untrained = validate(rn_untrained, val_loop)
# Plot Precision, Recall, and F1-score by Class for untrained model
plot_precision_recall_f1(rn_untrained, val_loop, 'Untrained')


# # Print model parameters
total_params = lambda x: sum(p.numel() for p in x.parameters())
print("Total number of parameters untrained model:", total_params(rn_untrained))
print("Total number of parameters trained model:", total_params(rn_pretrained))

