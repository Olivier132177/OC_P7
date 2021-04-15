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
#application_train=pd.read_csv(path+'application_train.csv')

#feature engeneering
deja_fait=True
if not deja_fait:
    fc.feature_engineering(path,True,True,True)
print('tout fini')
##################### suppression des variables inutiles ################################""
application_final=pd.read_csv(path+'application_final.csv', index_col=0).set_index('SK_ID_CURR')
application_final.loc[application_final['DAYS_EMPLOYED']>0,'DAYS_EMPLOYED']=np.nan
#application_final=application_final.drop(index=10990)
application_final
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
            'FLOORSMAX_AVG','FLOORSMAX_MODE', 'YEARS_BEGINEXPLUATATION_MEDI', 'TOTALAREA_MODE']

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
application_final
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



# ###########""

tests=False
if tests:
    application_final.columns
    #Test des différents hyper-paramètres
    meth=['Class_weight'] #['Aucune','Class_weight','SMOTE', 'RandomUnderSampler']
    algo=['LR'] 
    paramc=[0.01,0.1,1,10]
    df_resultats,_,_=fc.modelisation2(df_final_train,y_train,df_final_test,y_test,meth,algo,paramc,False)

    df_resultats
    #############

    df_resultats.to_csv(path+'df_resultatsN.csv')

############## modelisation avec des paramètres sélectionnés ######################
lr2=LogisticRegression(max_iter=2000, class_weight='balanced',random_state=0, C=0.01)
lr2.fit(df_final_train, y_train)
y_pred=lr2.predict(df_final_test)
y_prob=lr2.predict_proba(df_final_test).T[1]
df_final_test_pred=X_test.copy()
df_final_test_pred['y_pred']=y_prob
df_final_test_pred.to_csv(path+'df_pour_dashboard.csv')

coefs=lr2.coef_

df_coef=pd.concat([pd.Series(tab_nom_col),pd.Series(coefs[0])],axis=1)
df_coef.columns=['Nom_colonne','Coef']
df_coef['AbsCoef']=np.abs(df_coef['Coef'])
df_coef=df_coef.sort_values('AbsCoef',ascending=False)
df_coef=df_coef.sort_values('Coef',ascending=False)

df_coef.iloc[-60:]
df_coef
########## Meilleurs résultats obtenus ############
#Méthode : Class_weight C : 0.01
#Matrice de confusion :[[42262 18518]
# [ 1590  3502]]
#Accuracy : 0.695 ROC AUC : 0.758 AUC Precision-Recall : 0.237 F1 : 0.258

#étude des coefficients

resultatsf=pd.read_csv(path+'resultats_finaux.csv')
resultatsf
df_coef.to_csv(path+'coefficients.csv')

len(tab_nom_col)

df_final_train.columns=tab_nom_col
df_final_train['DAYS_EMPLOYED'].describe()

application_final['DAYS_EMPLOYED'].sort_values()

ddd=application_final[application_final['DAYS_EMPLOYED']>0]
ddd['DAYS_EMPLOYED'].sort_values()
ddd['NAME_INCOME_TYPE'].value_counts()

application_final[['TARGET','AMT_INCOME_TOTAL']].sort_values('AMT_INCOME_TOTAL')

application_final.loc[10990]

X_test