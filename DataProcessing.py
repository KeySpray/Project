from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load the Falah/Alzheimer_MRI dataset
train_dataset = load_dataset('Falah/Alzheimer_MRI', split='train')
test_dataset = load_dataset('Falah/Alzheimer_MRI', split='test')

# Print the number of examples and the first few samples
print("Number of examples:", len(train_dataset))
print(len(test_dataset))
print("Sample data:")
for example in train_dataset[:5]:
    print(example)

# Process Training Data 
images = train_dataset["image"]
input_training_vectors = []
train_labels = train_dataset["label"]
for image in images:
    input_training_vectors.append(np.array(image).flatten() / 255)
print(max(input_training_vectors[0]))
print(train_labels[0])

# Process Test Data
images = test_dataset["image"]
input_test_vectors = []
test_labels = test_dataset["label"]
for image in images:
    input_test_vectors.append(np.array(image).flatten() / 255)
print(max(input_test_vectors[0]))
print(test_labels[0])

