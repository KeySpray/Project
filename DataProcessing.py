from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models, transforms
import numpy as np
import torch

# Load the Falah/Alzheimer_MRI dataset for Logistic Regression
train_dataset = load_dataset('Falah/Alzheimer_MRI', split='train')
test_dataset = load_dataset('Falah/Alzheimer_MRI', split='test')

# Load and preprocess dataset for neural nets
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def transform_images(dataset, preprocess):
    images = [preprocess(image["image"]) for image in dataset]
    labels = torch.tensor([label["label"] for label in dataset])
    # Stack images into a single tensor
    return torch.stack(images), labels

# Transform both the training and test datasets
X_train_nn, Y_train_nn = transform_images(train_dataset, preprocess)
X_test_nn, Y_test_nn = transform_images(test_dataset, preprocess)

# Create TensorDatasets
train_data_nn = TensorDataset(X_train_nn, Y_train_nn)
test_data_nn = TensorDataset(X_test_nn, Y_test_nn)

# Create DataLoaders
train_loader_nn = DataLoader(train_data_nn, batch_size=32, shuffle=True)
test_loader_nn = DataLoader(test_data_nn, batch_size=32, shuffle=False)

# Print the number of examples and the first few samples
# print("Number of examples:", len(train_dataset))
# print("Sample data:")
# for example in train_dataset[:5]:
#     print(example)

flattened_image_size = 128 * 128

# Create empty train feature vectors, extract labels
X_train_lr = np.empty([len(train_dataset), flattened_image_size])
Y_train_lr = train_dataset["label"]

# Create empty test feature vectors, extract labels
X_test_lr = np.empty([len(test_dataset), flattened_image_size])
Y_test_lr = test_dataset["label"]

# Flatten and normalize the pixels between [0, 1]
for i, example in enumerate(train_dataset):
    X_train_lr[i] = np.array(example["image"]).flatten() / 255.0
# print(max(X_train_lr[0]))
# print(Y_train_lr[0])

# Flatten and normalize the pixels between [0, 1]
for i, example in enumerate(test_dataset):
    X_test_lr[i] = np.array(example["image"]).flatten() / 255.0
print(np.sum(x == 1 for x in Y_test_lr))
# print(max(X_test_lr[0]))
# print(Y_test_lr[0])

def fetch_data():
    return X_train_lr, Y_train_lr, X_test_lr, Y_test_lr, train_loader_nn, test_loader_nn