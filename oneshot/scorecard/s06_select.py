
def var_select_vif(vifcut):
    from datetime import datetime
    startTime = datetime.now()
    
    # Libraries
    import pandas as pd
    from patsy import dmatrices
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # Inputs
    trainbinfile = "/media/dennis/Internal Storage/PythonScripts/move/datafiles/trainbin.csv"
    varrankfile = "/media/dennis/Internal Storage/PythonScripts/move/datafiles/varrank.csv"
    viffile2 = "/media/dennis/Internal Storage/PythonScripts/move/datafiles/vif2.csv"
    
    # Imports
    trainbin = pd.read_csv(trainbinfile)
    varrank = pd.read_csv(varrankfile)
    
    varlist = [i for i in varrank.loc[varrank['rank'] <= 90]["variable"]]
    
    yvar = "moveind"
    xvar = varlist
    
    y = trainbin[yvar]
    X = trainbin[xvar]
    
    #gather features
    features = "+".join(varlist)
    
    # get y and X dataframes based on this regression:
    y, X = dmatrices(yvar + ' ~ ' + features, trainbin, return_type='dataframe')
    
    # For each X, calculate VIF and save in dataframe
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["features"] = X.columns
    
    vif.round(1)
    
    vif.to_csv(viffile2,index=False)
    
    print(datetime.now() - startTime)
