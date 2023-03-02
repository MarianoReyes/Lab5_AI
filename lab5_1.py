import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

data = pd.read_csv('dataset_phishing.csv')
# info del dataset
# print(data.head())
# print(data.dtypes.to_dict())
# print(data.describe())
# print(data.info())

''' INICIO DE LIMPIADO DE DATOS '''
# encoding de url
data['protocol'] = data['url'].apply(lambda x: x.split(':')[0])
data['domain'] = data['url'].apply(lambda x: x.split('/')[2])
data['path'] = data['url'].apply(lambda x: '/'.join(x.split('/')[3:]))

encoder = LabelEncoder()
data['protocol'] = encoder.fit_transform(data['protocol'])
data['domain'] = encoder.fit_transform(data['domain'])
data['path'] = encoder.fit_transform(data['path'])

data.drop(columns=['url'], inplace=True)

# print(data['status'].value_counts())

X = data.drop(columns=['status'])
y = data['status']

# para balancear aplicamos SMOTE
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)

data_res = pd.concat([X_res, y_res], axis=1)

X = data_res.drop(columns=['status'])
X = np.absolute(X)
y = data_res['status']

# SelectKBest
# print("SKB")
selector = SelectKBest(chi2, k=10)
selector.fit(X, y)

selected_cols = X.columns[selector.get_support()].tolist()

# guardar archivo limpiado en un archivo CSV
data_res[selected_cols + ['status']].to_csv('dataset_phishing_cleaned.csv')

'''
La métrica de desempeño principal que utilizaré es la precisión (accuracy), 
que se define como la proporción de predicciones correctas en relación al 
total de predicciones. La razón por la que elijo esta métrica es porque en 
este problema de clasificación binaria, es importante que el modelo tenga una 
alta tasa de aciertos en la clasificación de sitios web phishing y no 
phishing para evitar caer en falsos positivos (clasificar un sitio web 
legítimo como phishing) o falsos negativos (clasificar un sitio web phishing 
como legítimo), ya que ambos casos pueden tener consecuencias graves en 
términos de seguridad informática. La precisión nos da una idea clara de la 
tasa de aciertos del modelo en general y es fácil de interpretar. 
'''

''' FIN DE LIMPIADO DE DATOS '''

''' TASK 1.1 '''


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
print('Accuracy:', accuracy)


plt.scatter(X_test['domain'], X_test['path'], c=y_pred)
plt.xlabel('Domain')
plt.ylabel('Path')
plt.show()
