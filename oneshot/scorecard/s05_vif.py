from datetime import datetime
startTime = datetime.now()

# Libraries
import pandas as pd
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Inputs
trainbinfile = "/media/dennis/Internal Storage/PythonScripts/move/datafiles/trainbin.csv"
viffile = "/media/dennis/Internal Storage/PythonScripts/move/datafiles/vif.csv"

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

#gather features
features = "+".join(X.columns)

# get y and X dataframes based on this regression:
y, X = dmatrices(yvar + ' ~ ' + features, trainbin, return_type='dataframe')

# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

vif.round(1)

vif.to_csv(viffile,index=False)

print(datetime.now() - startTime)
