import random
import csv
from sklearn.manifold import TSNE
import math
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

print(data['status'].value_counts())

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
print("SKB")
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