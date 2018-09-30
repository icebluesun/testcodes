from datetime import datetime
startTime = datetime.now()

# Additional Research
#https://stackoverflow.com/questions/37906210/applying-pandas-qcut-bins-to-new-data

# Libraries
import pandas as pd
#from sklearn.model_selection import train_test_split
import gc
import numpy as np

# Collect Memory
gc.collect()

# Inputs
rawfile = "/media/dennis/Internal Storage/PythonScripts/move/datafiles/raw.csv"
trainfile = "/media/dennis/Internal Storage/PythonScripts/move/datafiles/train.csv"
testfile = "/media/dennis/Internal Storage/PythonScripts/move/datafiles/test.csv"
trainbinfile = "/media/dennis/Internal Storage/PythonScripts/move/datafiles/trainbin.csv"
"""
# Imports
raw = pd.read_csv(rawfile)

# Prep files for splitting
y = raw[["move"]]
X = raw.drop(columns=["move"])

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1979)

# Rejoining Train and Test
train = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

# Export for later use
train.to_csv(trainfile,index=False)
test.to_csv(testfile,index=False)
"""
# Imports
train = pd.read_csv(rawfile)

# Var List Prep - User MUST give a list of variables to do bins
idlist = ["load_year","load_month","append_year","append_month","number_of_businesses","moveind","move","move_state","moverate","zcta"]
varlist = list(train.columns.values)
varlist2 = [i for i in varlist if i not in idlist and "1312" not in i]

# MACRO START
binraw = train.copy()
cutoff=np.percentile(binraw["moverate"],90, axis=0)
binraw["moveind"]=[1 if i > cutoff  else 0 for i in binraw["moverate"]]

# Clear Memory because the next step takes too much memory for laptop to do
# del X, X_test, X_train, raw, y, y_test, y_train
gc.collect()

# Loop through list to create bins
for i in varlist2:
    var=i
    print(var+" start")
    dec="dec_"+var
    lab="lab_"+var
    groups=20
    try:
        binraw[dec] = pd.qcut(binraw[var],groups,labels=False, duplicates='drop')
        binraw[dec].fillna(-1,inplace=True)
        binraw[lab] = pd.qcut(binraw[var],groups, duplicates='drop')
    except:
        binraw[dec] = [(1-int(pd.isnull(i)==True))-1 for i in binraw[var]]
        binraw[lab] = binraw[var]
#        pass
    print(var+" done")
# MACRO END

# EXPORT
binraw.to_csv(trainbinfile,index=False)

print(datetime.now() - startTime)
