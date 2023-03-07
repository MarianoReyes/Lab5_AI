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

# Función de kernel gaussiano (RBF)
def gaussian_kernel(X, Y, gamma=1):
    dist_sq = np.sum(X**2, axis=1, keepdims=True) + np.sum(Y**2, axis=1) - 2 * X @ Y.T
    return np.exp(-gamma * dist_sq)

# Matriz de kernel gaussiano para los datos de entrenamiento
K_train = gaussian_kernel(X_train, X_train)

# Algoritmo de descenso de gradiente para entrenar el SVM con la matriz de kernel
def svm_train_kernel(K, y, C, learning_rate, num_epochs):
    n = K.shape[0]
    alpha = np.zeros(n)
    for epoch in range(num_epochs):
        hinge_loss = np.maximum(0, 1 - y * (K @ alpha))
        cost = C * np.sum(hinge_loss) / n + 0.5 * alpha.T @ (K @ alpha)
        grad = -C * (y * K.T) @ (hinge_loss > 0) / n + K @ alpha
        alpha = alpha - learning_rate * grad
    return alpha

# Entrenar el modelo de SVM con la matriz de kernel
C = 1
learning_rate = 0.01
num_epochs = 1000
alpha = svm_train_kernel(K_train, y_train, C, learning_rate, num_epochs)

# Evaluar el modelo en el conjunto de validación
K_validate = gaussian_kernel(X_validate, X_train)
y_pred = np.sign(K_validate @ alpha)
accuracy = np.mean(y_pred == y_validate)
print("Accuracy en validación:", accuracy)

# Evaluar el modelo en el conjunto de prueba
K_test = gaussian_kernel(X_test, X_train)
y_pred = np.sign(K_test @ alpha)
accuracy = np.mean(y_pred == y_test)
print("Accuracy en prueba:", accuracy)

# Graficar los grupos encontrados (usando las dos primeras variables)
K_plot = gaussian_kernel(X_test[:, :2], X_train[:, :2])
y_plot = np.sign(K_plot @ alpha)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_plot)
plt.show()
