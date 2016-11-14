#generate data
import pandas as pd
import numpy as np
np.random.seed(123)

variables = ['X','Y','Z']
labels = ['ID_0','ID_1','ID_2','ID_3','ID_4']
X = np.random.random_sample([5,3])*10
df = pd.DataFrame(X,columns = variables,index = labels)
df

#calculate the distance matrix
from scipy.spatial.distance import pdist,squareform
row_dist = pd.DataFrame(squareform(pdist(df,metric='euclidean')),columns = labels,index = labels)
row_dist

#apply complete linkage agglomeration to clusters
from scipy.cluster.hierarchy import linkage
row_clusters = linkage(pdist(df,metric = 'euclidean'),
                       method='complete')

pd.DataFrame(row_clusters,
             columns = ['row label 1','row label 2','distance','no. of items in clust.'],
             index = ['cluster %d' % (i+1) for i in range(row_clusters.shape[0])])

#visulaize the results in the form of a dendrogram
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
row_dendr = dendrogram(row_clusters,labels = labels)
plt.tight_layout()
plt.ylabel('Eucliden distance')
plt.show()


#attching dendrograms to a heat map
#1
fig = plt.figure(figsize=(8,8))
axd = fig.add_axes([0.09,0.1,0.2,0.6])
row_dendr = dendrogram(row_clusters, orientation='left')

#2
df_rowclust = df.ix[row_dendr['leaves'][::-1]]

#3
axm = fig.add_axes([0.23,0.1,0.6,0.6])
cax = axm.matshow(df_rowclust,interpolation='nearest', cmap='hot_r')

#4
axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))
plt.show()