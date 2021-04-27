from numpy import random
from numpy.core.shape_base import block
import pandas as pd
from sklearn import impute
from sklearn.utils import class_weight
import fonctions as fc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,auc,f1_score\
    ,roc_auc_score, precision_recall_curve, plot_roc_curve\
        , plot_confusion_matrix, plot_precision_recall_curve,auc
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from imblearn.over_sampling import SMOTE,RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler,ClusterCentroids
from sklearn.ensemble import RandomForestClassifier
from joblib import dump,load



pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 150)

path='/home/olivier/Desktop/openclassrooms/P7/data/'

#bureau_balance=pd.read_csv(path+'bureau_balance.csv')
#credit_card_balance=pd.read_csv(path+'credit_card_balance.csv')
#info_colonnes=pd.read_csv(path+'HomeCredit_columns_description.csv', encoding='cp850', index_col=0)
#installments_payments=pd.read_csv(path+'installments_payments.csv')
#pos_cash_balance=pd.read_csv(path+'POS_CASH_balance.csv')
#previous_application=pd.read_csv(path+'previous_application.csv')
#sample_submission=pd.read_csv(path+'sample_submission.csv')
#bureau=pd.read_csv(path+'bureau.csv')
#application_test=pd.read_csv(path+'application_test.csv')
application_train=pd.read_csv(path+'application_train.csv')
application_train['TARGET'].value_counts(normalize=True)
#feature engeneering
deja_fait=True
if not deja_fait:
    fc.feature_engineering(path,True,True,True)
print('tout fini')
##################### suppression des variables inutiles ################################""
application_final=pd.read_csv(path+'application_final.csv', index_col=0).set_index('SK_ID_CURR')
application_final.loc[application_final['DAYS_EMPLOYED']>0,'DAYS_EMPLOYED']=np.nan
#application_final=application_final.drop(index=10990)

application_final['LN_REVENU_TOTAL']=np.log(application_final['AMT_INCOME_TOTAL'])

variables_supprimees=['AGE_RANGE','APARTMENTS_MEDI','YEARS_BUILD_MODE']#,'SK_ID_CURR']

variables_supprimees_2=['CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'HOUR_APPR_PROCESS_START',
            'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'FLAG_EMAIL', 'FLAG_PHONE',
            'FLAG_OWN_REALTY', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
            'REG_CITY_NOT_WORK_CITY', 'OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
            'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_YEAR', 
            'COMMONAREA_MODE', 'NONLIVINGAREA_MODE', 'ELEVATORS_MODE', 'NONLIVINGAREA_AVG',
            'FLOORSMIN_MEDI', 'LANDAREA_MODE', 'NONLIVINGAREA_MEDI', 'LIVINGAPARTMENTS_MODE',
            'FLOORSMIN_AVG', 'LANDAREA_AVG', 'FLOORSMIN_MODE', 'LANDAREA_MEDI',
            'COMMONAREA_MEDI', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'BASEMENTAREA_AVG',
            'BASEMENTAREA_MODE', 'NONLIVINGAPARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 
            'LIVINGAPARTMENTS_AVG', 'ELEVATORS_AVG', 'YEARS_BUILD_MEDI', 'ENTRANCES_MODE',
            'NONLIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'LIVINGAPARTMENTS_MEDI',
            'YEARS_BUILD_MODE', 'YEARS_BEGINEXPLUATATION_AVG', 'ELEVATORS_MEDI', 'LIVINGAREA_MEDI',
            'YEARS_BEGINEXPLUATATION_MODE', 'NONLIVINGAPARTMENTS_AVG', 'HOUSETYPE_MODE',
            'FONDKAPREMONT_MODE', 'EMERGENCYSTATE_MODE'   ,'OWN_CAR_AGE','CAR_TO_EMPLOYED_RATIO', 
            'CAR_TO_BIRTH_RATIO', 'EXT_SOURCES_PROD', 'APARTMENTS_AVG', 'APARTMENTS_MODE',
            'APARTMENTS_MEDI','ENTRANCES_AVG', 'ENTRANCES_MEDI', 'LIVINGAREA_AVG', 'FLOORSMAX_MEDI', 
            'FLOORSMAX_AVG','FLOORSMAX_MODE', 'YEARS_BEGINEXPLUATATION_MEDI', 'TOTALAREA_MODE',
            'EXT_SOURCE_1_y','EXT_SOURCE_2_y','EXT_SOURCE_3_y','DAYS_BIRTH_y']

meilleures_variables=['TARGET','INCOME_TO_BIRTH_RATIO','AMT_INCOME_TOTAL',
'ORGANIZATION_TYPE','AMT_GOODS_PRICE','AMT_CREDIT','CREDIT_TERM','FLAG_DOCUMENT_13',
'FLAG_DOCUMENT_3','NAME_EDUCATION_TYPE','FLAG_DOCUMENT_14','FLAG_DOCUMENT_6',
'CREDIT_LENGTH','EXT_SOURCE_3_x','FLAG_DOCUMENT_16','EXT_SOURCES_VAR','EXT_SOURCES_MAX',
'DEBT_CREDIT_RATIO','OCCUPATION_TYPE','NAME_TYPE_SUITE','FLAG_DOCUMENT_18',
'CODE_GENDER','FLAG_DOCUMENT_5','NAME_INCOME_TYPE','FLAG_DOCUMENT_8','OCCUPATION_TYPE',
'NAME_HOUSING_TYPE','FLAG_DOCUMENT_9','FLAG_DOCUMENT_15']

ajout=False
if ajout:
    application_final=application_final[meilleures_variables]
elif not ajout:
    application_final=application_final.loc[:,np.isin(application_final.columns,variables_supprimees, invert=True)]
    application_final=application_final.loc[:,np.isin(application_final.columns,variables_supprimees_2, invert=True)]
colonnes_retenues=application_final.columns

#séparation du train set et du test set, recuperation des noms de colonnes
train_set,test_set,label,col_cat,col_num,features = fc.preparation_df(application_final)

#train test split du train set
X_train, X_test, y_train, y_test = train_test_split(train_set[features],\
     train_set['TARGET'], test_size=0.25, random_state=0, stratify=train_set['TARGET'])


#imputation des valeurs manquantes
col_num_train, col_cat_train=fc.imputation_valeurs_manquantes(X_train,col_num, col_cat)
col_num_test, col_cat_test=fc.imputation_valeurs_manquantes(X_test,col_num, col_cat)

#preprocessing
cat_col_cat = [col_cat_train[column].unique()\
     for column in col_cat_train] #libellé des colonnes catégories
df_final_train, ohe_train,ss_train=fc.preprocessing(col_num_train,col_cat_train, cat_col_cat)
df_final_test,ohe_test,ss_test=fc.preprocessing(col_num_test,col_cat_test,cat_col_cat)

#récupèration du nom des colonnes du df train final
tab_nom_col_cat,tab_nom_col=fc.nom_colonnes(col_cat_train,col_num_train)

df_final_test.columns=tab_nom_col
df_final_train.columns=tab_nom_col

df_final_train.to_csv(path+'df_final_train.csv')
df_final_test.to_csv(path+'df_final_test.csv')
y_test.to_csv(path+'y_test.csv')
y_train.to_csv(path+'y_train.csv')

##############"
df_final_test=pd.read_csv(path+'df_final_test.csv', index_col=0)
df_final_train=pd.read_csv(path+'df_final_train.csv', index_col=0)
y_test=pd.read_csv(path+'y_test.csv', index_col=0)
y_train=pd.read_csv(path+'y_train.csv', index_col=0)


np.sort(df_final_test.columns.to_list())

# ###########""

tests=False
if tests:
    #Test des différents hyper-paramètres
    meth=['Class_weight','SMOTE', 'RandomUnderSampler']
    algo=['RF','LR'] 
    paramc=[0.01,0.1,1,10]
    parammd=[7,10,13,16]
    #meth=['Aucune','Class_weight','SMOTE', 'RandomUnderSampler']
    #algo=['RF','LR'] 
    #paramc=[0.01,0.1,1,10]
    #parammd=[7,8,9,10,11,12,13]
    df_resultats_2=fc.modelisation2(df_final_train,y_train,df_final_test,y_test,meth,algo,paramc,parammd,False)
    df_resultats_2['F_beta']=(5*df_resultats_2['Precision']*df_resultats_2['Recall'])\
    /((4*df_resultats_2['Precision'])+df_resultats_2['Recall'])
    df_resultats_2.to_csv(path+'df_resultatsF1B.csv')

resultats_finaux=pd.read_csv(path+'df_resultatsF1B.csv')
resultats_finaux
############## modelisation avec des paramètres sélectionnés ######################
lr2=LogisticRegression(max_iter=2000, class_weight='balanced',random_state=0, C=0.01)
lr2.fit(df_final_train, y_train)
y_pred=lr2.predict(df_final_test)
y_prob=lr2.predict_proba(df_final_test).T[1]

dump(lr2, 'modele_sauvegarde.joblib') 

acc, mat, a_u_c, f1, auc_pr=fc.scores(y_test,y_pred,y_prob,lr2,df_final_test, graphs=True)

df_final_test_pred=X_test.copy()
#df_final_test_pred['y_pred']=y_prob
df_final_test_pred.to_csv(path+'df_test_avant_transfo_pour_dashboard.csv') #pour les graphs/stats

df_final_test.index=X_test.index

df_final_test.to_csv(path+'df_test_prep_pour_dashboard.csv') #pour la modélisation


df_train_pour_dash=X_train.copy()
df_train_pour_dash['label']=y_train
df_train_pour_dash.to_csv(path+'df_train_pour_dashboard.csv') #pour les graphs/stats

X_test

y_train
coefs=lr2.coef_
df_coef=pd.concat([pd.Series(tab_nom_col),pd.Series(coefs[0])],axis=1)
df_coef.columns=['Variables','Coef']
df_coef['AbsCoef']=np.abs(df_coef['Coef'])
df_coef=df_coef.set_index('Variables')
#df_coef.index==df_final_test.columns
df_coef.to_csv(path+'coefficients.csv')

inter=lr2.intercept_[0]

#398172
#412932
#135480

iden=412932

pred=df_final_test_pred.loc[iden,'y_pred']
inter_coef=df_coef[['Coef']].join(df_final_test.loc[iden])
inter_coef.columns=['Coef','Value']
inter_coef['impact']=inter_coef['Coef']*inter_coef['Value']
impa=inter_coef['impact'].sum()
np.sort(inter_coef.index)
inter_coef.sort_values('Coef').head(15)
inter
print(pred,'',impa)

impa #-0.94 somme des coefs
inter #-0.74 intercept
impa+inter # log de la cote du label "1" :-1.68
np.exp(impa+inter) #cote du label "1" 0.19
# coef : contribution au log du rapport de cotes
np.exp(impa) #0.39
np.exp(inter) #0.48
np.exp(impa)*np.exp(inter) #0.19

pred #0.16
pred/(1-pred) #cote : 0.19
#pred=0.19*(1-pred)
#pred=0.19-pred*0.19
np.log(pred/(1-pred)) #log du rapport de cotes -1.68


df_minmaxmoy=pd.concat([test_set[col_num].mean(),test_set[col_num].std(),test_set[col_num].min(),
test_set[col_num].max(),test_set[col_num].median()],axis=1)
df_minmaxmoy.columns=['moyenne','ecart_type','minimum','maximum','mediane']
df_minmaxmoy.to_csv(path+'df_minmaxmoy.csv')

########## Meilleurs résultats obtenus ############
#Méthode : Class_weight C : 0.01
#Matrice de confusion :[[42254 18526]
# [ 1567  3525]]
#Accuracy : 0.695 ROC AUC : 0.76 AUC Precision-Recall : 0.241 F1 : 0.26

df_resultats_MN=pd.read_csv(path+'df_resultatsMN.csv', index_col=0)
df_resultats_MN.sort_values(['algo','methode'])

df_resultats_MN['numer_F1']=df_resultats_MN['Recall']*df_resultats_MN['Precision']*2
df_resultats_MN['denom_F1']=df_resultats_MN['Recall']+df_resultats_MN['Precision']

df_resultats_MN[['algo', 'methode', 'Best_params',  'Confusion_matrix',
       'Precision_Recall_AUC', 'Recall',
       'Precision', 'numer_F1','denom_F1','F1_score']].sort_values('F1_score')


df_resultats_MN.sort_values(['F_beta'])