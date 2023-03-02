import numpy as np
import pandas as pd
import csv
import random
import math
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

''' KNN SIN LIBRERIAS'''

print("\nTask 1.1 del Laboratorio 5, implementación de KNN (K-Nearest Neighbors)")
# Cargamos el conjunto de datos


def load_dataset(filename, split):
    dataset = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Ignoramos las filas que contengan valores vacíos o no numéricos
            if line.strip() == '':
                continue
            row = line.strip().split(',')
            for i in range(len(row)-1):
                if not is_number(row[i]):
                    break
                row[i] = float(row[i])
            else:
                row[-1] = str(row[-1])
                dataset.append(row)
    train_size = int(len(dataset) * split)
    train_set = dataset[:train_size]
    test_set = dataset[train_size:]
    return train_set, test_set

# Función auxiliar para verificar si un valor es numérico


def is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

# Definimos la métrica de desempeño


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual))

# Calculamos la distancia Euclidiana entre dos puntos


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return math.sqrt(distance)

# Calculamos la distancia Euclidiana entre un punto de prueba y todos los puntos del conjunto de entrenamiento


def get_neighbors(train, test_row, num_neighbors):
    distances = []
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda x: x[1])
    neighbors = []
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

# Predecimos la etiqueta de un punto de prueba según la clase mayoritaria entre los k vecinos más cercanos


def predict_classification(train, test_row, num_neighbors):
    # Obtenemos los k vecinos más cercanos
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    # Predecimos la etiqueta según la clase mayoritaria
    prediction = max(set(output_values), key=output_values.count)
    return prediction

# Implementación del algoritmo de K-Nearest Neighbors


def k_nearest_neighbors(train, test, num_neighbors):
    predictions = []
    for test_row in test:
        output = predict_classification(train, test_row, num_neighbors)
        predictions.append(output)
    return predictions


# Cargamos el conjunto de datos y lo dividimos en entrenamiento y prueba
filename = 'dataset_phishing_cleaned.csv'
split = 0.8
training_set, test_set = load_dataset(filename, split)
print(
    f'\nTraining={len(training_set)} Test={len(test_set)}')

# Evaluamos el algoritmo de K-Nearest Neighbors con k=3
predictions = k_nearest_neighbors(training_set, test_set, 3)
actual = [row[-1] for row in test_set]
accuracy = accuracy_metric(actual, predictions)
print('\Presición sin librerías:', accuracy)

# Obtener las etiquetas de predicción para el conjunto de prueba
predictions = k_nearest_neighbors(training_set, test_set, 3)

# Crear una lista con los valores de las características X y Y para cada punto en el conjunto de prueba
x = [row[0] for row in test_set]
y = [row[1] for row in test_set]

# Crear una lista con las etiquetas de predicción correspondientes
colors = ['red' if label == '1' else 'blue' for label in predictions]

# Crear el gráfico de dispersión
plt.scatter(x, y, c=colors)

# Agregar etiquetas al gráfico
plt.title('Etiquetas de predicción')
plt.xlabel('Característica X')
plt.ylabel('Característica Y')

# Mostrar el gráfico
plt.savefig('knn_sinlibs.jpg')
plt.show()


''' KNN CON LIBRERIAS'''
# Cargar el dataset
data = pd.read_csv('dataset_phishing_cleaned.csv', index_col=0)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2, random_state=42)


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X.to_numpy()]
        return np.array(predicted_labels)

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train)
                     for x_train in self.X_train.to_numpy()]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train.iloc[i] for i in k_indices]
        most_common_label = max(set(k_nearest_labels),
                                key=k_nearest_labels.count)
        return most_common_label


knn = KNN(k=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Presición con librerías:', accuracy)


plt.scatter(X_test['domain'], X_test['path'], c=y_pred)
plt.xlabel('Dominio')
plt.ylabel('Path')
plt.savefig('knn_conlibs.jpg')
plt.show()
