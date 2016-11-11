# use all variables in the dataset and train a multiple regression model:
#load data Housing Dataset: https://archive.ics.uci.edu/ml/datasets/Housing
import pandas as pd
#df = pd.read_csv('https://archive.ics.uci.edu/ml/datasets/housing/Housing.data',header = None,sep = '\s+')
df = pd.read_csv('F:/developSamples/ml/housing.data',header = None,sep = '\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
X = df.iloc[:,:-1].values
y = df['MEDV'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=0)

slr = LinearRegression()
slr.fit(X_train,y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

#2.	Plot the residuals(the differences or vertical distances between the 
#actual and predicted values) versus the predicted values to diagnose our regression model.
import matplotlib.pyplot as plt
plt.scatter(y_train_pred, y_train_pred - y_train,
c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test,
c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()

#another useful quantitative measure of a model's preformance is the so-called Mean Squared Error(MSE)
from sklearn.metrics import mean_squared_error
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),mean_squared_error(y_test, y_test_pred)))

#R^2

from sklearn.metrics import r2_score 
print('R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))