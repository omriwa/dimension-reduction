import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get dataset
dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:, -1].values

# Split dataset
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Use PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

