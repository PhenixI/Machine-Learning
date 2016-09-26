import os
os.chdir('E:\\books_and_materials\\machine learning\\machine-learning\\6-Perceptron and Neural Networks\\Perceptron_python')
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header = None)
import matplotlib.pyplot as plt
import numpy as np
y = df.iloc[0:100,4].values
y= np.where(y=='Iris-setosa',-1,1)
X = df.iloc[0:100,[0,2]].values
import Perceptron
ppn = Perceptron.Perceptron(eta=0.1,n_iter=10)
ppn.fit(X,y)
import DecisionBoundary as DB
DB.plot_decision_regions(X,y,classifier=ppn)
