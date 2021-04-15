from inspect import TPFLAGS_IS_ABSTRACT
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix,accuracy_score,auc,f1_score\
    ,roc_auc_score, precision_recall_curve, plot_roc_curve\
        , plot_confusion_matrix, plot_precision_recall_curve,auc\
    ,recall_score, precision_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import fonctions as fc
path='/home/olivier/Desktop/openclassrooms/P7/data/'

df_final_test=pd.read_csv(path+'df_final_test.csv', index_col=0)
df_final_train=pd.read_csv(path+'df_final_train.csv', index_col=0)
y_test=pd.read_csv(path+'y_test.csv', index_col=0)
y_train=pd.read_csv(path+'y_train.csv', index_col=0)


depth=8
arbres=1000
methode='U'  # U S N W
grid=False
param_X={'max_depth':[6,7,8,9,10,11,12]}
seuil=np.arange(0,101,1)

if methode=='S':
    smot=SMOTE(random_state=0)
    df_train_2, y_train_2 = smot.fit_resample(np.array(df_final_train), np.array(y_train))
elif methode=='U':
    rus = RandomUnderSampler(random_state=0)
    df_train_2, y_train_2 = rus.fit_resample(np.array(df_final_train), np.array(y_train))
else :
    df_train_2=df_final_train
    y_train_2=y_train
if methode=='W':
    weight='balanced'
else:
    weight=None

rf=RandomForestClassifier(n_estimators=arbres,random_state=0,class_weight=weight, max_depth=depth)
if not grid:
    rf.fit(df_train_2,y_train_2)
    y_pred=rf.predict_proba(df_final_test)
    importances=rf.feature_importances_
    est=rf
else:
    gs=GridSearchCV(rf,param_X,scoring='roc_auc',cv=3, verbose=3)
    gs.fit(df_train_2,y_train_2)
    y_pred=gs.predict_proba(df_final_test)
    est=gs
tab_res,df_resu,auc,auc_pr=fc.score_par_seuil(y_test,y_pred,seuil)
tab_res
df_resu.sort_index().iloc[:50]
df_resu.sort_index().iloc[50:]

auc
plot_roc_curve(est,df_final_test,y_test)
plt.show()
plot_precision_recall_curve(est,df_final_test,y_test)
plt.show()
print(f1_score(y_test,[round(x) for x in y_pred.T[1]]))
confusion_matrix(y_test,[round(x) for x in y_pred.T[1]])
re=recall_score(y_test,[round(x) for x in y_pred.T[1]])
pr=precision_score(y_test,[round(x) for x in y_pred.T[1]])

re
pr

(2*re*pr)/(re+pr)
auc

3409/(3409+1683)
3409/(3409+18435)

df_resu.loc[69]








std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(df_final_train[:n].shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the impurity-based feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(df_final_train[:n].shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(df_final_train[:n].shape[1]), indices)
plt.xlim([-1, df_final_train[:n].shape[1]])
plt.show()



