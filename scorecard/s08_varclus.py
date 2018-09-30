from datetime import datetime
startTime = datetime.now()

# Additional Research
#https://stackoverflow.com/questions/37906210/applying-pandas-qcut-bins-to-new-data

# Libraries
import pandas as pd

from numba import vectorize, cuda
@vectorize(['float32(float32, float32)'], target='cuda')

# Inputs
trainbinfile = "/media/dennis/Internal Storage/PythonScripts/move/datafiles/trainbin.csv"
varclusfile = "/media/dennis/Internal Storage/PythonScripts/move/datafiles/varclus.csv"

# Imports
trainbin = pd.read_csv(trainbinfile)

# Var List Prep - User MUST give a list of variables to do bins
varlist = [i for i in list(trainbin.columns.values) if i[:4] == 'dec_']

# needed imports
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

X = trainbin[varlist].T
Z = linkage(X, 'ward')

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

c, coph_dists = cophenet(Z, pdist(X))
c

# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

maxdist = max([Z[i][2] for i in range(len(Z))])
distbin = 25

from scipy.cluster.hierarchy import fcluster
max_d = 4*maxdist/distbin
clusters = fcluster(Z, max_d, criterion='distance')
numclust = max(clusters)

import numpy as np
test = np.concatenate((varlist,clusters),)

clusterlist = pd.DataFrame({'cluster':clusters,'variable':varlist})

clusterlist.to_csv(varclusfile,index=False)
# MACRO START
print(datetime.now() - startTime)

# raw 0:00:09.575596
# cuda 0:00:06.641165