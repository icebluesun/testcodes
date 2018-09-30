def varselect_vif(vifcutoff):
    from datetime import datetime
    startTime = datetime.now()
    
    # Libraries
    import pandas as pd
    from patsy import dmatrices
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # Inputs
    trainbinfile = "/media/dennis/Internal Storage/PythonScripts/move/datafiles/trainbin.csv"
    viffile3 = "/media/dennis/Internal Storage/PythonScripts/move/datafiles/vif3.csv"
    
    # Imports
    trainbin = pd.read_csv(trainbinfile)
    vif3 = pd.read_csv(viffile3)
    
    varlist = [i for i in vif3.loc[vif3['VIF Factor'] <= vifcutoff]["features"] if i != 'Intercept']
    
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
    print(vif.loc[vif['features'] != 'Intercept']['VIF Factor'].max())
    
    vif.to_csv(viffile3,index=False)
    
    print(datetime.now() - startTime)

varselect_vif(2)