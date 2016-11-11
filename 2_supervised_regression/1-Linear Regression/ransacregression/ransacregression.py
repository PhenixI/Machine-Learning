#load data Housing Dataset: https://archive.ics.uci.edu/ml/datasets/Housing
import pandas as pd
#df = pd.read_csv('https://archive.ics.uci.edu/ml/datasets/housing/Housing.data',header = None,sep = '\s+')
df = pd.read_csv('F:/developSamples/ml/housing.data',header = None,sep = '\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()

X = df['RM'].values.reshape(-1,1)
y = df['MEDV'].values.reshape(-1,1)

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import RANSACRegressor
#By setting the residual_threshold parameter to 5.0, we
#only allowed samples to be included in the inlier set if their vertical distance to the
#fitted line is within 5 distance units, which works well on this particular dataset. 

ransac = RANSACRegressor(LinearRegression(),
                         max_trials=100,
                         min_samples=50,
                         residual_metric=lambda x: np.sum(np.abs(x), axis=1),
                         residual_threshold=5.0,
                         random_state=0)
ransac.fit(X,y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask],c='blue', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],c='lightgreen', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper left')
plt.show()

print('Slope: %.3f' % ransac.estimator_.coef_[0])
print('Intercept: %.3f' % ransac.estimator_.intercept_)