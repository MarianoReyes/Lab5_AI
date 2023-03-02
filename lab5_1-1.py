import numpy as np
import pandas as pd
import csv
import random
import math
import matplotlib.pyplot as plt

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
print('\nAccuracy sin libs:', accuracy)

''' KNN CON LIBRERIAS'''
