import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the Iris dataset from a CSV file
file_path = 'iris.data'  # Replace with your CSV file path
df = pd.read_csv(file_path)

print("Dataset Sample:")
print(df.head())

if 'species' in df.columns:
    X = df.drop('species', axis=1)
    y = df['species']

    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    y_pred = encoder.inverse_transform(y_pred)
    y_test = encoder.inverse_transform(y_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
else:
    print("Error: 'species' column not found in the dataset.")
