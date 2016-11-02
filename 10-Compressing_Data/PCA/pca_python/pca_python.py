#pca test
#load data
import pandas as pd
import numpy as np
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)

#separate training and test sets, standardize it to unit variance
from sklearn.cross_validation import train_test_split
X,y = df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 0)
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

#detial 

#cov_mat = np.cov(X_train_std.T)
#eigen_vals,eigen_vecs = np.linalg.eig(cov_mat)
#print('\nEigenvalues \n %s' % eigen_vals)

#variance explained ratio
#tot = sum(eigen_vals)
#var_exp = [(i / tot) for i in
#sorted(eigen_vals, reverse=True)]
#cum_var_exp = np.cumsum(var_exp)
#import matplotlib.pyplot as plt
#plt.bar(range(1,14), var_exp, alpha=0.5, align='center',
#label='individual explained variance')
#plt.step(range(1,14), cum_var_exp, where='mid',
#label='cumulative explained variance')
#plt.ylabel('Explained variance ratio')
#plt.xlabel('Principal components')
#plt.legend(loc='best')
#plt.show()

# sorting the eigenpairs by decreasing order of the eigenvalues:
#eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
#eigen_pairs.sort(reverse=True)

#print('\nEigenvalues \n %s' % eigen_pairs)

#collect the two eignevectors that correspond to the two largest values to capture about 60 percent of the variance in this dataset
#w = np.hstack((eigen_pairs[0][1][:,np.newaxis],
#               eigen_pairs[1][1][:,np.newaxis]))

#print('Matrix W: \n',w)

#transform dataset onto PCA subsapce
#X_train_pca = X_train_std.dot(w)

import pca
X_train_pca = pca.pca(X_train_std,2)

#visualize the transformed Wine training set
import matplotlib.pyplot as plt
colors = ['r','b','g']
markers = ['s','x','o']
for l,c,m in zip(np.unique(y_train),colors,markers):
    plt.scatter(X_train_pca[y_train == l,0],X_train_pca[y_train==l,1],c=c,label = l,marker = m)
plt.xlabel('PC 1')
plt.xlabel('PC 2')
plt.legend(loc = 'lower left')
plt.show()



