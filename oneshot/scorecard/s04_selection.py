from datetime import datetime
startTime = datetime.now()

# Libraries
import pandas as pd

# Inputs
trainbinfile = "/media/dennis/Internal Storage/PythonScripts/move/datafiles/trainbin.csv"
binsumfile = "/media/dennis/Internal Storage/PythonScripts/move/datafiles/binsum.csv"
varrankfile = "/media/dennis/Internal Storage/PythonScripts/move/datafiles/varrank.csv"
viffile = "/media/dennis/Internal Storage/PythonScripts/move/datafiles/vif.csv"

# Imports
trainbin = pd.read_csv(trainbinfile)
binsum = pd.read_csv(binsumfile)
varrank = pd.read_csv(varrankfile)
vif = pd.read_csv(viffile)

print(datetime.now() - startTime)
