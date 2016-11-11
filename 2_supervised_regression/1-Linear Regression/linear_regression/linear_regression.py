#load data Housing Dataset: https://archive.ics.uci.edu/ml/datasets/Housing
import pandas as pd
#df = pd.read_csv('https://archive.ics.uci.edu/ml/datasets/housing/Housing.data',header = None,sep = '\s+')
df = pd.read_csv('F:/developSamples/ml/housing.data',header = None,sep = '\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()

#visualize the pair-wise correlations between the different features
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid',context = 'notebook')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols],size = 2.5)
plt.show()

#compute coorelation and use heatmap to visualize it
import numpy as np
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,cbar = True,annot = True,square = True,fmt = '.2f',annot_kws = {'size':15},yticklabels = cols,xticklabels = cols)
plt.show() 

#train model and predict

import os
os.chdir('E:/machine-learning/2_supervised_regression/1-Linear Regression/linear_regression')

X = df['RM'].values.reshape(-1,1)
y = df['MEDV'].values.reshape(-1,1)
from sklearn.preprocessing import StandardScaler
from LinearRegressionGD import LinearRegressionGD
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y)
lr = LinearRegressionGD()
lr.fit(X_std,y_std)

plt.plot(range(1,lr.n_iter+1),lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()
