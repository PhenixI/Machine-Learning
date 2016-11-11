#load data Housing Dataset: https://archive.ics.uci.edu/ml/datasets/Housing
import pandas as pd
#df = pd.read_csv('https://archive.ics.uci.edu/ml/datasets/housing/Housing.data',header = None,sep = '\s+')
df = pd.read_csv('F:/developSamples/ml/housing.data',header = None,sep = '\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()

X = df['RM'].values.reshape(-1,1)
y = df['MEDV'].values.reshape(-1,1)

from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X,y)

print ('Slope: %.3f' % slr.coef_[0])
print ('Intercept: %.3f' % slr.intercept_)

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')
    return None

lin_regplot(X,y,slr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()
