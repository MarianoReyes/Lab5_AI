import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar datos desde el archivo CSV
data = pd.read_csv("dataset_phishing_cleaned.csv")

# Dividir los datos en entrenamiento, validación y prueba
train, validate, test = np.split(
    data.sample(frac=1, random_state=42), [int(0.8 * len(data)), int(0.9 * len(data))]
)

# Preparar los datos
X_train = train.iloc[:, :-1].values
y_train = np.where(train.iloc[:, -1].values == "phishing", -1, 1)
X_validate = validate.iloc[:, :-1].values
y_validate = np.where(validate.iloc[:, -1].values == "phishing", -1, 1)
X_test = test.iloc[:, :-1].values
y_test = np.where(test.iloc[:, -1].values == "phishing", -1, 1)

# Normalizar los datos
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean) / std
X_validate = (X_validate - mean) / std
X_test = (X_test - mean) / std

# Función de costo del SVM
def svm_cost(w, X, y, C):
    n = X.shape[0]
    z = y * (X @ w)
    hinge_loss = np.maximum(0, 1 - z)
    cost = C * np.sum(hinge_loss) / n + 0.5 * np.sum(w**2)
    return cost


# Gradiente de la función de costo del SVM
def svm_grad(w, X, y, C):
    n = X.shape[0]
    z = y * (X @ w)
    hinge_loss_grad = -y * (z < 1)
    grad = C * (X.T @ hinge_loss_grad) / n + w
    return grad


# Algoritmo de descenso de gradiente para entrenar el SVM
def svm_train(X, y, C, learning_rate, num_epochs):
    n, d = X.shape
    w = np.zeros(d)
    for epoch in range(num_epochs):
        cost = svm_cost(w, X, y, C)
        grad = svm_grad(w, X, y, C)
        w = w - learning_rate * grad
    return w


# Entrenar el modelo de SVM
C = 1
learning_rate = 0.01
num_epochs = 1000
w = svm_train(X_train, y_train, C, learning_rate, num_epochs)

# Evaluar el modelo en el conjunto de validación
y_pred = np.sign(X_validate @ w)
accuracy = np.mean(y_pred == y_validate)
print("Accuracy en validación:", accuracy)

# Evaluar el modelo en el conjunto de prueba
y_pred = np.sign(X_test @ w)
accuracy = np.mean(y_pred == y_test)
print("Accuracy en prueba:", accuracy)

# Graficar los grupos encontrados (usando las dos primeras variables)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)
plt.show()
