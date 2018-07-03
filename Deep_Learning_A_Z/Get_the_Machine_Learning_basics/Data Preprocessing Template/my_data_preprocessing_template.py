# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 21:22:01 2018

@author: Shashwat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the Dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values