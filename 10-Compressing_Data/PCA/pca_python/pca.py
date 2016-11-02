import numpy as np



def loadDataSet(fileName,delim='\t'):
    fr=open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr=[map(float,line) for line in stringArr]
    return mat(datArr)

def replaceNanWithMean():
    datMat = loadDataSet('secom.data',' ')
    numFeat= shape(datMat)[1]
    for i in range(numFeat):
        meanVal= mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i])
        datMat[nonzero(isnan(datMat[:,i].A))[0],i]=meanVal
    return datMat

def pca(dataMat,topNfeat=9999999):
    #1.standarization
#    meanVlas=mean(dataMat,axis=0)
#    meanRemoved = dataMat-meanVlas
#   2. compute covariance matrix
    covMat=np.cov(dataMat.T)
    #3.get eigenvalues and eigenvectors
    eigVals,eigVects = np.linalg.eig(covMat)
    #print('\nEigenvalues \n %s' % eigVals)

    #4.sort the eigenpairs by descending order of the eigenvalues
    eigen_pairs = [(np.abs(eigVals[i]),eigVects[:,i]) for i in range(len(eigVals))]
    eigen_pairs.sort(reverse=True)

    #print('\nEigenpairs \n %s' % eigen_pairs)
    #5. select k eigenvectors (trade-off between computational efficiency and the performance of the classifier)
    redEigVects= np.hstack((eigen_pairs[i][1][:,np.newaxis] for i in range(topNfeat)))

    #6 transform data onto the PCA subsapce
    #lowDDataMat = dataMat * redEigVects
    lowDDataMat = dataMat.dot(redEigVects)

    #reconMat = (lowDDataMat * redEigVects.T) +meanVlas
    return lowDDataMat #,reconMat

