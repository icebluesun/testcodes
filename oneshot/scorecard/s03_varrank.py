from datetime import datetime
startTime = datetime.now()

# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

# Inputs
trainbinfile = "/media/dennis/Internal Storage/PythonScripts/move/datafiles/trainbin.csv"
varrankfile = "/media/dennis/Internal Storage/PythonScripts/move/datafiles/varrank.csv"

# Imports
trainbin = pd.read_csv(trainbinfile)

varlist = list(trainbin.columns.values)
varlist2 = [i for i in varlist if i[:4] in ("dec_")]

yvar = "moveind"
xvar = varlist2
nvar = len(xvar)
nobs = len(trainbin)

y = trainbin[yvar]
X = trainbin[xvar]

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=nvar,random_state=0)

forest.fit(X, y)

importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)

indices = np.argsort(importances)[::-1]

# Create Variable Rank
varrank = pd.DataFrame({"col": range(X.shape[1])})
varrank["rank"] = varrank["col"]+1
varrank["feature"] = indices[varrank["col"]]
varrank["variable"] = [varlist2[indices[i]] for i in varrank["col"]]
varrank["score"]=importances[indices[varrank["col"]]]
varrank =varrank.drop(columns=["col"])

varrank.to_csv(varrankfile,index=False)

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

for f in range(X.shape[1]):
    print( "Rank %d. Feature %d, %s (%f)" % ( f+1 , indices[f] , varlist2[indices[f]] , importances[indices[f]] ) )


# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

print(datetime.now() - startTime)
