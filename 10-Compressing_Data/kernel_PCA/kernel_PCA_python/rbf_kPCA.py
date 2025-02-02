#kernel PCA, kernel :rbf (radix basis function)
from scipy.spatial.distance import pdist,squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def rbf_kernel_pca(X,gamma,n_components):
    """
    RBF kernel PCA implementation
    Parameters
    --------------
    X: {NumPy ndarry},shape = [n_samples, n_features]
    gamma: float 
       Tuning parameter of the RBF kernel
    n_components: int
       Number of principal components to return

    Returns
    --------------
    X_pc: {NumPy ndarray}, shape=[n_samples,k_features]
      Projected dataset
    lambdas: list
      Eigenvalues
    """
    #Calculate pairwise squared Euclidean distances in the MxN dimensional dataset
    sq_dists = pdist(X,'sqeuclidean')

    #Convert pairwise distances into a square matrix
    mat_sq_dists = squareform(sq_dists)

    #Compute the symmetric kernel matrix
    K = exp(-gamma * mat_sq_dists)

    #center the kernel matrix
    N = K.shape[0]
    one_n = np.ones((N,N))/N
    K = K - one_n.dot(K)-K.dot(one_n) + one_n.dot(K).dot(one_n)

    #Obtaining eigenpairs from the centered kernel matrix
    #numpy.eigh returns them in sorted order
    eigvals,eigvecs = eigh(K)

    #Collect the top k eigenvectors (projected samples)
    alphas = np.column_stack((eigvecs[:,-i] for i in range(1,n_components + 1)))

    #Collect the corresponding eigenvalues
    lambdas = [eigvals[-i] for i in range(1,n_components +1)]

    return alphas, lambdas

def project_x(x_new,X,gamma,alphas,lambdas):
    pair_dist = np.array([np.sum((x_new-row)**2) for row in X])
    k = np.exp(-gamma*pair_dist)

    return k.dot(alphas/lambdas)