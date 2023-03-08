import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load data from CSV file
data = pd.read_csv("dataset_phishing_cleaned.csv")

# Drop rows with missing values
data = data.dropna()

# Remove duplicates
data = data.drop_duplicates()

# Split data into training, validation, and test sets
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Prepare the data
X_train = train.iloc[:, :-1].values
y_train = np.where(train.iloc[:, -1].values == "phishing", -1, 1)
X_test = test.iloc[:, :-1].values
y_test = np.where(test.iloc[:, -1].values == "phishing", -1, 1)

# Normalize the training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM model
gamma = 0.1
C = 1
clf = SVC(kernel="rbf", gamma=gamma, C=C)
clf.fit(X_train, y_train)

# Evaluate the SVM model on the test set
y_test_pred = clf.predict(X_test)

# Calculate the accuracy of the model on the test set
accuracy = np.mean(y_test_pred == y_test)
print("\nAccuracy on the test set:", accuracy)

# Plot the groups and support vectors (using the first two variables)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred)
plt.scatter(
    clf.support_vectors_[:, 0],
    clf.support_vectors_[:, 1],
    s=100,
    facecolors="none",
    edgecolors="k",
)
plt.show()
