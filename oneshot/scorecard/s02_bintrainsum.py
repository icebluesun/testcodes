from datetime import datetime
startTime = datetime.now()

# Libraries
import pandas as pd
import math as m

# Inputs
trainbinfile2 = "/media/dennis/Internal Storage/PythonScripts/move/datafiles/trainbin2.csv"
binsumfile2 = "/media/dennis/Internal Storage/PythonScripts/move/datafiles/binsum2.csv"
ivfile2 = "/media/dennis/Internal Storage/PythonScripts/move/datafiles/iv2.csv"

# Imports
trainbin = pd.read_csv(trainbinfile2)
trainbin["counts"] = 1

# Var List Prep - User MUST give a list of variables to do bins
varlist = list(trainbin.columns.values)
varlist2 = [i for i in varlist if i[:4] in ("dec_")]

idlist = ["load_year","load_month","append_year","append_month","number_of_businesses","move","moveind","move_state","moverate","zcta","counts"]
varlist = list(trainbin.columns.values)
varlist2 = [i for i in varlist if i not in idlist and "1312" not in i and i[:4] not in ("dec_","lab_")]

# WOE ADJUST
woeadj = 0.5

# TOTAL
counts = trainbin["counts"].sum()
events = trainbin["moveind"].sum()
nevent = counts-events

for var in varlist2:
    print('Doing Var = '+var)
    
    if trainbin["dec_"+var].min() == -1:
        
        test1=trainbin.loc[trainbin["dec_"+var] == -1]
        test2=trainbin.loc[trainbin["dec_"+var] > -1]
        
        test1a = pd.DataFrame(test1.groupby(["dec_"+var])[["move","moveind", "number_of_businesses", "counts"]].sum())
        test1a["group"]=test1a.index
        test1a["range"]="Missing"
        test1a["moverate_overall"]=test1a["move"]/test1a["number_of_businesses"]
        test1a["moverate_average"]=pd.DataFrame(test1.groupby(["dec_"+var])[["moverate"]].mean())
        test1a["moveindrate"]=test1a["moveind"]/test1a["counts"]

        test2a = pd.DataFrame(test2.groupby(["dec_"+var,"lab_"+var])[["move","moveind", "number_of_businesses", "counts"]].sum())
        test2a["group"]=[i[0] for i in test2a.index]
        test2a["range"]=[i[1] for i in test2a.index]
        test2a["moverate_overall"]=test2a["move"]/test2a["number_of_businesses"]
        test2a["moverate_average"]=pd.DataFrame(test2.groupby(["dec_"+var,"lab_"+var])[["moverate"]].mean())
        test2a["moveindrate"]=test2a["moveind"]/test2a["counts"]
    
        test3=test1a.append(test2a)
        
    else:
        
        test3=pd.DataFrame(trainbin.groupby(["dec_"+var,"lab_"+var])[["move","moveind", "number_of_businesses", "counts"]].sum())
        test3["group"]=[i[0] for i in test3.index]
        test3["range"]=[i[1] for i in test3.index]
        test3["moverate_overall"]=test3["move"]/test3["number_of_businesses"]
        test3["moverate_average"]=pd.DataFrame(trainbin.groupby(["dec_"+var,"lab_"+var])[["moverate"]].mean())
        test3["moveindrate"]=test3["moveind"]/test3["counts"]
    
    test3["var"]=var
    test3["events"]=test3["moveind"]
    test3["nevent"]=test3["counts"]-test3["moveind"]
    test3["pct_counts"]=test3["counts"]/counts
    test3["pct_events"]=(test3["events"])/events
    test3["pct_nevent"]=(test3["nevent"])/nevent
    test3["woe_pre"]=((test3["nevent"]+woeadj)/nevent)/((test3["events"]+woeadj)/events)
    test3["woe"]=[m.log(i) for i in test3["woe_pre"]]
    test3["iv"]=test3["woe"]*(test3["pct_nevent"]-test3["pct_events"])
    
    if var == varlist2[0]:
        test4 = test3
    else:
        test4 = test4.append(test3)

    print('Finishing Var = '+var)
    
test4 = test4[["var","group","range","counts","moveind","moveindrate","number_of_businesses","move","moverate_overall","moverate_average","woe","iv"]]
test4.to_csv(binsumfile2,index=False)

test5 = pd.DataFrame(test4.groupby(["var"])[["iv"]].sum())
test5["var"] = test5.index
test5.to_csv(ivfile2,index=False)

print(datetime.now() - startTime)
