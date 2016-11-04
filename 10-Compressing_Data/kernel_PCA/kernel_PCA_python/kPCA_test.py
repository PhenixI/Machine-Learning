import matplotlib.pyplot as plt
import numpy as np
#a. separating half-moon data
#1.creating a two-dimensional dataset of 100 sample points representing two half-moon shapes:
from sklearn.datasets import make_moons
X,y = make_moons(n_samples= 100,random_state = 123)
plt.scatter(X[y==0, 0], X[y==0, 1],color='red', marker='^', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1],color='blue', marker='o', alpha=0.5)
plt.show()

#2.using standard PCA
from sklearn.decomposition import PCA
scikit_pca = PCA(n_components = 2)
X_spca = scikit_pca.fit_transform(X)
fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))
ax[0].scatter(X_spca[y==0, 0], X_spca[y==0, 1],color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y==1, 0], X_spca[y==1, 1],color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_spca[y==0, 0], np.zeros((50,1))+0.02,color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y==1, 0], np.zeros((50,1))-0.02,color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()

#3.using rbf kernel PCA
from matplotlib.ticker import FormatStrFormatter
import rbf_kPCA as rbf
X_kpca = rbf.rbf_kernel_pca(X,gamma=15,n_components=2)
fig,ax = plt.subplots(nrows = 1,ncols = 2,figsize= (7,3))
ax[0].scatter(X_kpca[y==0,0],X_kpca[y==0,1],color = 'red',marker = '^',alpha = 0.5)
ax[0].scatter(X_kpca[y==1,0],X_kpca[y==1,1],color = 'blue',marker='o',alpha = 0.5)
ax[1].scatter(X_kpca[y==0,0],np.zeros((50,1))+0.02,color='red',marker = '^',alpha = 0.5)
ax[1].scatter(X_kpca[y==1,0], np.zeros((50,1))-0.02,color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
plt.show()


#b. separating concentric circles
#1. data generation
from sklearn.datasets import make_circles
X,y = make_circles(n_samples=1000,random_state = 123, noise = 0.1,factor = 0.2)
plt.scatter(X[y==0, 0], X[y==0, 1],color='red', marker='^', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1],color='blue', marker='o', alpha=0.5)
plt.show()

#2.using standard PCA
scikit_pca = PCA(n_components = 2)
X_spca = scikit_pca.fit_transform(X)

fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))
ax[0].scatter(X_spca[y==0, 0], X_spca[y==0, 1],color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y==1, 0], X_spca[y==1, 1],color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_spca[y==0, 0], np.zeros((500,1))+0.02,color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y==1, 0], np.zeros((500,1))-0.02,color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()

#using rbf kernel PCA
X_kpca = rbf.rbf_kernel_pca(X,gamma=15,n_components=2)

fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))
ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1],color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_kpca[y==0, 0], np.zeros((500,1))+0.02,color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y==1, 0], np.zeros((500,1))-0.02,color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()

#project new data points
import rbf_kPCA as rbf
from sklearn.datasets import make_moons
X,y = make_moons(n_samples= 100,random_state = 123)
alphas,lambdas = rbf.rbf_kernel_pca(X,gamma=15,n_components=1)

#assume the 26th point is a new data
x_new = X[25]
x_new
x_proj = alphas[25]# original projection
x_reproj = rbf.project_x(x_new,X,gamma=15,alphas=alphas,lambdas=lambdas)



