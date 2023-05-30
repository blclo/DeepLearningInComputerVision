#from src.data.dataloader import LatentDataset
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import numpy as np
import os
import torch

'''
train_dataset = LatentDataset(train=True)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = LatentDataset(train=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
'''
latent_codes_dir_with_sunglasses = '/work3/s212725/GANs/data/with_sunglasses'
latent_codes_dir_without_sunglasses = '/work3/s212725/GANs/data/without_sunglasses'

# Get latent codes from directory containing them and iterate through them
latent_codes = []
labels = []

for latent_code in os.listdir(latent_codes_dir_with_sunglasses):
    latent_code = torch.load(os.path.join(latent_codes_dir_with_sunglasses, latent_code))
    latent_codes.append(latent_code)
    labels.append(1) # with sunglasses

for latent_code in os.listdir(latent_codes_dir_without_sunglasses):
    latent_code = torch.load(os.path.join(latent_codes_dir_without_sunglasses, latent_code))
    latent_codes.append(latent_code)
    labels.append(0) # without sunglasses

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(latent_codes, labels, test_size=0.2, random_state=42)

# Step 3: Preprocess the data if needed (e.g., scaling, normalization)

# Step 4: Train the SVM model
svm_classifier = svm.SVC(kernel='linear')  # Create an SVM classifier with linear kernel
svm_classifier.fit(X_train, y_train)  # Fit the classifier to the training data

# Step 5: Evaluate the model
y_pred = svm_classifier.predict(X_test)  # Predict labels for the test data
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
print("Accuracy:", accuracy)

# Get the support vectors and corresponding coefficients
support_vectors = svm_classifier.support_vectors_
coefficients = svm_classifier.coef_[0]

# Extract the latent direction from the coefficients
latent_direction = np.dot(support_vectors.T, coefficients)

# Normalize the latent direction (optional)
latent_direction /= np.linalg.norm(latent_direction)
