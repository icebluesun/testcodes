from datetime import datetime
startTime = datetime.now()

# Libraries
import pandas as pd

# Inputs
rawfile = "/media/dennis/Internal Storage/PythonScripts/move/datafiles/raw.csv"

# Imports
raw = pd.read_csv(rawfile)
ycut = raw["moverate"].quantile(.9)
raw["moveind"] = (raw["moverate"] >= ycut).astype(int)

cols_all = list(raw.columns)
cols_yvr = ["moveind"]
cols_ids = ["zcta","load_year","load_month","number_of_businesses","move","move_state","moverate"]
cols_xvr = [i for i in cols_all if i not in cols_yvr and i not in cols_ids and "1412" not in i]

y = raw[cols_yvr]
x = raw[cols_xvr]

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer

num_pipeline = Pipeline([
        ('imputer', Imputer(strategy='median')),
        ('std_scaler', StandardScaler()),
    ])

print(datetime.now() - startTime)
