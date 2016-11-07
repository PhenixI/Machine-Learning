#k-fold cross-validation

#read in the dataset
import pandas as pd
#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',header=None)
df = pd.read_csv('E:/machine-learning/data/wdbc.data',header=None)

from sklearn.preprocessing import LabelEncoder
X=df.loc[:,2:].values
y = df.loc[:,1].values
lc = LabelEncoder()
y = lc.fit_transform(y)

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

#Combining transformers and estimators in a pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe_lr = Pipeline([('scl',StandardScaler()),('pca',PCA(n_components=2)),('clf',LogisticRegression(random_state=1))])
#pipe_lr.fit(X_train,y_train)
#print('Test Accuracy: %.3f' % pipe_lr.score(X_test,y_test))

#stratifiedKFold
import numpy as np
from sklearn.cross_validation import StratifiedKFold
kfold = StratifiedKFold(y=y_train,n_folds = 10,random_state = 1)
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k+1,np.bincount(y_train[train]), score))

#use k-fold cross-validation
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(estimator = pipe_lr,X=X_train,y=y_train,cv=10,n_jobs=-1)
print ('CV accuracy scores: %s' % scores)
print ('CV accuracy: %.3f +/- %.3f' %(np.mean(scores),np.std(scores)))

