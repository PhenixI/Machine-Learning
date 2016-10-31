

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

#Encoding class labels
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

#performing one-hot encoding on nominal features

#a.
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0])
ohe.fit_transform(X).toarray()

#b
pd.get_dummies(df[['price', 'color', 'size']])





