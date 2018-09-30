# Time Program
from datetime import datetime
startTime = datetime.now()

# Libraries
import pandas as pd

# Inputs
#trainbinfile = "/media/dennis/Internal Storage/PythonScripts/move/datafiles/trainbin.csv"
trainbinfile = "/media/dennis/Internal Storage/PythonScripts/move/datafiles/trainbin.csv"
newbinfile = "/media/dennis/Internal Storage/PythonScripts/move/datafiles/newbin.csv"
trainbinfile2 = "/media/dennis/Internal Storage/PythonScripts/move/datafiles/trainbin2.csv"

# Imports
trainbin = pd.read_csv(trainbinfile)
newbin = pd.read_csv(newbinfile)

newbin["change"]=1-(newbin["group"]==newbin["newbin"])
newbin.head()

varsum=pd.DataFrame(newbin.groupby(["var"])[["change"]].sum())
varsum

deplist = ["zcta","move","number_of_businesses","moveind"]
varlist = [i for i, j in zip(varsum.index,varsum["change"]) if j > 0]
declist = ["dec_"+i for i in varlist]
keepvar = deplist+declist

trainbin2 = trainbin[keepvar]

for var in varlist:
    k1 =newbin.loc[newbin['var'].isin([var]),["var","group","newbin"]]
    k1["dec_"+var]=k1["group"]
    k1["dec2_"+var]=k1["newbin"]
    k2=k1[["dec_"+var,"dec2_"+var]]
    
    #k2.head()
    #trainbin2.head()
    
    trainbin2 = pd.merge(trainbin2, k2, on="dec_"+var, how='outer')
    #trainbin3.head()

print(datetime.now() - startTime)

trainbin2.drop(columns=declist,inplace=True)

trainbin2.to_csv(trainbinfile2,index=False)

"""
### k1 =newbin.loc[newbin['var'].isin(varlist),["var","group","newbin"]]
#k1["dec2_pct_owns_ind_1yrate"]=k1["newbin"]
#k1["dec_pct_owns_ind_1yrate"]=k1["group"]
k1.head()

trainbin2 = trainbin[["zcta","move","number_of_businesses","pct_owns_ind_1yrate","dec_pct_owns_ind_1yrate"]]

trainbin3 = pd.merge(trainbin2, k1, on='dec_pct_owns_ind_1yrate', how='outer')

trainbin3.head()

"""