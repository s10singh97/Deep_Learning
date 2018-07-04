# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 16:38:54 2018

@author: Shashwat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the Dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the Data into Training Set and Test Set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
