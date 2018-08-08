#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:09:09 2018

@author: mithilesh.abhishek
"""

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#working with missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# working with categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncode_x = LabelEncoder()
labelEncode_x.fit(X[:, 0])
X[:, 0] = labelEncode_x.fit_transform(X[:, 0])
onehotencode_x = OneHotEncoder(categorical_features = [0])
X = onehotencode_x.fit_transform(X).toarray()

labelEncode_y = LabelEncoder()
labelEncode_y.fit(y)
y = labelEncode_y.fit_transform(y)


# spliting the dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)



