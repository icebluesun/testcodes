def varrank_tree(df_x,df_y):

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import ExtraTreesClassifier
    
    obs = df_y.count()
    minobs = np.int(obs/25)
    nest = np.max([np.min([250,obs/50]).astype(int),20]).astype(int)
    
    forest = ExtraTreesClassifier(
            n_estimators=nest,
            min_samples_split=minobs,
            random_state=1979
            )
    forest.fit(df_x, df_y)
    importance = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
    indices = np.argsort(importance)[::-1]
    
    varlistrank = pd.DataFrame(df_x.columns,columns=["feature_name"])
    varlistrank["score"]=importance
    
    return varlistrank.sort_values(by=["score"],ascending=False).reset_index(drop=True)

def varclus_agglo(df_x):

    import numpy as np
    import pandas as pd
    from sklearn import cluster

    obs = len(df_x.columns)
    
    agglo = cluster.FeatureAgglomeration(n_clusters=int(np.sqrt(obs)))
    agglo.fit(df_x)
    
    varclus_agglo = pd.DataFrame(df_x.columns,columns=["feature_name"])
    varclus_agglo["cluster"]=agglo.labels_
    
    return varclus_agglo.sort_values(by=["cluster"],ascending=True).reset_index(drop=True)
