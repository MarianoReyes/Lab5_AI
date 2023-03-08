import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv("dataset_phishing_cleaned.csv")

# Drop rows with missing values
data = data.dropna()

# Remove duplicates
data = data.drop_duplicates()

# Split data into training, validation, and test sets
train, validate, test = np.split(
    data.sample(frac=1, random_state=42), [int(0.8 * len(data)), int(0.9 * len(data))]
)

# Prepare the data
X_train = train.iloc[:, :-1].values
y_train = np.where(train.iloc[:, -1].values == "phishing", -1, 1)
X_validate = validate.iloc[:, :-1].values
y_validate = np.where(validate.iloc[:, -1].values == "phishing", -1, 1)
X_test = test.iloc[:, :-1].values
y_test = np.where(test.iloc[:, -1].values == "phishing", -1, 1)

# RBF kernel function
def kernel_rbf(X1, X2, gamma):
    dist = (
        np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
    )
    return np.exp(-gamma * dist)


# SVM cost function
def svm_cost(alpha, y, K, C):
    n = K.shape[0]
    margin = y * K @ alpha
    hinge_loss = np.maximum(0, 1 - margin)
    cost = C * np.sum(hinge_loss) / n + 0.5 * alpha.T @ K @ alpha
    return cost


# Gradient of the SVM cost function
def svm_grad(alpha, y, K, C):
    n = K.shape[0]
    margin = y * K @ alpha
    hinge_loss_grad = -y * (margin < 1)
    grad = C * K @ hinge_loss_grad + K @ alpha
    return grad


# Gradient descent algorithm for training the SVM
def svm_train(K, y, C, learning_rate, num_epochs):
    n = K.shape[0]
    alpha = np.zeros(n)
    batch_size = 10
    for epoch in range(num_epochs):
        if epoch %100 == 0: print("charging " + str((epoch/num_epochs) *100) + "%")

        for i in range(0, n, batch_size):
            X_batch = K[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            # Calcular el gradiente para el batch actual
            margin = y_batch * X_batch.dot(alpha)
            hinge_loss_grad = -y_batch * (margin < 1)
            grad = C * X_batch.T.dot(hinge_loss_grad) + alpha
            # Actualizar los pesos
            alpha = alpha - learning_rate * grad

        #cost = svm_cost(alpha, y, K, C)
    w = (alpha * y) @ X_train
    return alpha, w


# Evaluate the SVM model on a new dataset
def svm_predict(alpha, y, X_train, X_test, gamma):
    K_test = kernel_rbf(X_test, X_train, gamma)
    margin = K_test @ (alpha * y)
    return np.sign(margin)


# Normalize the training data
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean) / std

# Calculate the kernel matrix for the training set
gamma = 0.1
K_train = kernel_rbf(X_train, X_train, gamma)

# Train the SVM model
C = 1
learning_rate = 0.1
num_epochs = 1000
alpha, w = svm_train(K_train, y_train, C, learning_rate, num_epochs)

# Find the support vectors and the bias
support_vectors = X_train[np.abs(K_train @ alpha - 1) < 1e-4]
bias = np.mean(
    y_train[np.abs(K_train @ alpha - 1) < 1e-4]
    - K_train[np.abs(K_train @ alpha - 1) < 1e-4] @ (alpha * y_train)
)

# Evaluar el modelo de SVM en el conjunto de prueba
X_test = (X_test - mean) / std
K_test = kernel_rbf(X_test, X_train, gamma)
y_test_pred = svm_predict(alpha, y_train, X_train, X_test, gamma)

# Calcular la precisión del modelo en el conjunto de prueba
accuracy = np.mean(y_test_pred == y_test)
print("Precisión en el conjunto de prueba:", accuracy)


# Graficar los grupos encontrados y los vectores de soporte (usando las dos primeras variables)
y_pred = svm_predict(alpha, y_train, X_train, X_test, gamma)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)
plt.scatter(
    support_vectors[:, 0],
    support_vectors[:, 1],
    s=100,
    facecolors="none",
    edgecolors="k",
)
plt.show()
