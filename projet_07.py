import pandas as pd
import fonctions as fc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,auc,f1_score,roc_auc_score
pd.set_option('display.max_colwidth', 40)
pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 150)

path='/home/olivier/Desktop/openclassrooms/P7/data/'

bureau_balance=pd.read_csv(path+'bureau_balance.csv')
credit_card_balance=pd.read_csv(path+'credit_card_balance.csv')
info_colonnes=pd.read_csv(path+'HomeCredit_columns_description.csv', encoding='cp850', index_col=0)
installments_payments=pd.read_csv(path+'installments_payments.csv')
pos_cash_balance=pd.read_csv(path+'POS_CASH_balance.csv')
previous_application=pd.read_csv(path+'previous_application.csv')
sample_submission=pd.read_csv(path+'sample_submission.csv')
bureau=pd.read_csv(path+'bureau.csv')
application_test=pd.read_csv(path+'application_test.csv')
application_train=pd.read_csv(path+'application_train.csv')

########################### feature engeneering 1 #####################

feat_eng1=True
if not feat_eng1:
    bureau=pd.read_csv('bureau.csv')
    B=fc.create_bureau_features(bureau) #creation de 10 features
    B.to_csv(path+'bureau_avec_features.csv')
    print("fini")
else :
    bureau_avec_features=pd.read_csv(path+'bureau_avec_features.csv', index_col=0)

########################### feature engeneering 2 #####################
feat_eng2=True
if not feat_eng2:
    application_test=pd.read_csv(path+'application_test.csv')
    application_train=pd.read_csv(path+'application_train.csv')
    application=fc.feat_eng(application_train, application_test)
    application.to_csv(path+'application_all.csv')
else :
    application=pd.read_csv(path+'application_all.csv', index_col=0)

##################### Modelisation ################################""

feat_eng3=True
if not feat_eng3:
    B2=bureau_avec_features.iloc[:,-8:-6].join(bureau_avec_features.iloc[:,-5:]).join(bureau_avec_features.iloc[:,0])
    B2=B2.drop_duplicates()
    application_final=pd.merge(application, B2, on='SK_ID_CURR')
    application_final.to_csv(path+'application_all.csv')
else :
    application_final=pd.read_csv(path+'application_all.csv', index_col=0)


##################### classification des variables ################################""
#label
label='TARGET'
#features
features=application_final.columns[np.isin(application_final.columns,'TARGET', invert=True)]
#colonnes de catégories
col_cat=application_final.loc[:,features].select_dtypes(exclude='number').columns #colonnes de catégories
#colonnes_numeriques
col_oth=application_final.loc[:,features].select_dtypes(include='number').columns
#colonnes de booleens
col_boo=col_oth[col_oth.str.contains('FLAG')\
    |col_oth.str.contains('REGION_NOT')|\
        col_oth.str.contains('CITY_NOT')]
#colonnes_numeriques
col_num=col_oth[np.isin(col_oth, col_boo, invert=True)]


###########imputation des valeurs manquantes
sp=SimpleImputer(strategy='mean')
