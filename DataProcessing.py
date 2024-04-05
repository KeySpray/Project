from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load the Falah/Alzheimer_MRI dataset
train_dataset = load_dataset('Falah/Alzheimer_MRI', split='train')
test_dataset = load_dataset('Falah/Alzheimer_MRI', split='test')

flattened_image_size = 128 * 128

# Create empty train feature vectors, extract labels
input_training_vectors = np.empty([len(train_dataset, flattened_image_size)])
train_labels = train_dataset["label"]

# Create empty test feature vectors, extract labels
input_test_vectors = np.empty([len(test_dataset, flattened_image_size)])
test_labels = test_dataset["label"]

# Print the number of examples and the first few samples
print("Number of examples:", len(train_dataset))
print(len(test_dataset))
print("Sample data:")
for example in train_dataset[:5]:
    print(example)

# Process Training Data 
train_images = train_dataset["image"]

# Flatten and normalize the pixels between [0, 1]
for image in train_images:
    input_training_vectors.append(np.array(image).flatten() / 255)
print(max(input_training_vectors[0]))
print(train_labels[0])

# Process Test Data
test_images = test_dataset["image"]

# Flatten and normalize the pixels between [0, 1]
for image in test_images:
    input_test_vectors.append(np.array(image).flatten() / 255)
print(max(input_test_vectors[0]))
print(test_labels[0])

