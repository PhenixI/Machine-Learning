

#simple example data frame
import pandas as pd
from io import StringIO
csv_data = '''A,B,C,D
              1.0,2.0,3.0,4.0
              5.0,6.0,,8.0
              0.0,11.0,12.0,'''
df = pd.read_csv(StringIO(csv_data))

#return a number of missing values per column as follows:
df.isnull().sum()

#1.removing features or samples
df.dropna()
df.dropna(axis = 1)

#2.imputing missing values
from sklearn.preprocessing import Imputer
#other strategy can be median, and most_frequent
imr = Imputer(missing_values='NaN',strategy = 'mean',axis=0)
imr =imr.fit(df)
imputed_data = imr.transform(df.values)
imputed_data

#Handling categorical data

#generate data
import pandas as pd
df = pd.DataFrame([['green', 'M', 10.1, 'class1'],['red', 'L', 13.5, 'class2'],['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']
df

#1.mapping ordinal features
size_mapping = {'XL':3,'L':2,'M':1}
df['size'] = df['size'].map(size_mapping)
df

#2.Encoding class labels
import numpy as np
class_mapping = {label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
class_mapping

df['classlabel'] = df['classlabel'].map(class_mapping)
df

#inv_class_mapping = {v: k for k,v in class_mapping.items()}
#df['classlabel'] = df['classlabel'].map(inv_class_mapping)
#df

from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
y

#3.performing one-hot encoding on nominal features

#a.
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0])
ohe.fit_transform(X).toarray()

#b
pd.get_dummies(df[['price', 'color', 'size']])


#Partitioning a dataset in training and testsets

#dataset,Wine,https://archive.ics.uci.edu/ml/datasets/Wine
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
df_wine.columns = ['Class label', 'Alcohol',
                   'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']

#print('Class labels',np.unique(df_wine['Class label']))
#df_wine.head()

#1.convenient way
from sklearn.cross_validation import train_test_split
X,y = df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 0)


#Bringing features onto same scale

#1.normalization
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

#2.standardized
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)


#Selecting meaningful features
#1.Sparse solutions with L1 regularization
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty = 'l1',C=0.1)
lr.fit(X_train_std,y_train)
print("Training accurarcy: ",lr.score(X_train_std,y_train));
print("Test accurarcy: ",lr.score(X_test_std,y_test))
lr.intercept_
lr.coef_

#2.Sequential feature selection algorithms
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
knn = KNeighborsClassifier(n_neighbors=2)
import SBS
sbs = SBS.SBS(knn,k_features=1)
sbs.fit(X_train_std,y_train)

k_feat = [len(list(k)) for k in sbs.subsets_]
plt.plot(k_feat,sbs.scores_,marker='o')
plt.ylim([0.7,1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()

#3 assessing feature importance with random forests
from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators = 10000,random_state = 0,n_jobs = -1)
forest.fit(X_train,y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[f],importances[indices[f]]))

plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]),
importances[indices],
color='lightblue',
align='center')
plt.xticks(range(X_train.shape[1]),
feat_labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

