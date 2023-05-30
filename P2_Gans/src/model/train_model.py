from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os
import torch

latent_codes_dir_with_sunglasses = '/work3/s212725/GANs/data/with_sunglasses'
latent_codes_dir_without_sunglasses = '/work3/s212725/GANs/data/without_sunglasses'

# Get latent codes from directory containing them and iterate through them
latent_codes = []
labels = []

for latent_code_file in os.listdir(latent_codes_dir_with_sunglasses):
    latent_code_path = os.path.join(latent_codes_dir_with_sunglasses, latent_code_file)
    latent_code_data = np.load(latent_code_path)
    latent_code = latent_code_data['w']  # Assuming the key for latent codes is 'latent_code'
    latent_code = np.squeeze(latent_code)  # Remove the extra dimension
    latent_code = latent_code.flatten()  # Flatten the latent code
    latent_codes.append(latent_code)
    labels.append(1)  # with sunglasses

for latent_code_file in os.listdir(latent_codes_dir_without_sunglasses):
    latent_code_path = os.path.join(latent_codes_dir_without_sunglasses, latent_code_file)
    latent_code_data = np.load(latent_code_path)
    latent_code = latent_code_data['w']  # Assuming the key for latent codes is 'latent_code'
    latent_code = np.squeeze(latent_code)  # Remove the extra dimension
    latent_code = latent_code.flatten()  # Flatten the latent code
    latent_codes.append(latent_code)
    labels.append(0)  # without sunglasses

# Convert the latent codes and labels to numpy arrays
latent_codes = np.array(latent_codes)
print("Initial latent_codes shape: ", latent_codes.shape)
labels = np.array(labels)

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(latent_codes, labels, test_size=0.4, random_state=42)

# Convert the training data to torch tensors
X_train = torch.from_numpy(X_train).float()
print("Initial shape: ", X_train.shape)
y_train = torch.from_numpy(y_train)

# Reshape X_train to the desired shape (1, 18, 512)
X_train = X_train.view(X_train.shape[0], *X_train.shape[1:])


# Step 4: Train the SVM model
svm_classifier = svm.SVC(kernel='linear')  # Create an SVM classifier with linear kernel
svm_classifier.fit(X_train, y_train)  # Fit the classifier to the training data
print(X_train.shape)
print(y_train.shape)

# Step 5: Evaluate the model
X_test = torch.from_numpy(X_test).float()
X_test = X_test.view(X_test.shape[0], *X_test.shape[1:])
y_pred = svm_classifier.predict(X_test)  # Predict labels for the test data
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
print("Accuracy:", accuracy)

# Get the support vectors and corresponding coefficients
support_vectors = svm_classifier.support_vectors_
coefficients = svm_classifier.coef_[0]

print(coefficients.shape)
# Step 6: Normalize the weight vector to obtain the latent direction
latent_direction = coefficients / np.linalg.norm(coefficients)

print(latent_direction.shape)
# Specify the original shape to unflatten the array
original_shape = (1, 18, 512)

# Unflatten the array
unflattened_array = latent_direction.reshape(original_shape)
print(unflattened_array.shape)