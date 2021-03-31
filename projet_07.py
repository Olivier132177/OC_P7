import pandas as pd
from sklearn import impute
import fonctions as fc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,auc,f1_score,roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
    bureau=pd.read_csv(path+'bureau.csv')
    B=fc.create_bureau_features(bureau) #creation de 10 features
    B.to_csv(path+'bureau_avec_features.csv')
    print("fini")
    bureau_avec_features=pd.read_csv(path+'bureau_avec_features.csv', index_col=0)
else :
    bureau_avec_features=pd.read_csv(path+'bureau_avec_features.csv', index_col=0)

bureau_avec_features.columns
########################### feature engeneering 2 #####################
feat_eng2=False
if not feat_eng2:
    application_test=pd.read_csv(path+'application_test.csv')
    application_train=pd.read_csv(path+'application_train.csv')
    application=fc.feat_eng(application_train, application_test)
    application.to_csv(path+'application_all.csv')
    application=pd.read_csv(path+'application_all.csv', index_col=0)
else :
    application=pd.read_csv(path+'application_all.csv', index_col=0)

##################### Modelisation ################################""

feat_eng3=False
if not feat_eng3:
    B2=bureau_avec_features.iloc[:,-6:-4].join(bureau_avec_features.iloc[:,-3:]).join(bureau_avec_features.iloc[:,0])
    B2=B2.drop_duplicates()
    application_final=pd.merge(application, B2, on='SK_ID_CURR')
    application_final.to_csv(path+'application_final.csv')
    application_final=pd.read_csv(path+'application_final.csv', index_col=0)

else :
    application_final=pd.read_csv(path+'application_final.csv', index_col=0)


##################### classification des variables ################################""
application_final=pd.read_csv(path+'application_final.csv', index_col=0)
#label########################### feature engeneering 2 #####################
feat_eng2=False
if not feat_eng2:
    application_test=pd.read_csv(path+'application_test.csv')
    application_train=pd.read_csv(path+'application_train.csv')
    application=fc.feat_eng(application_train, application_test)
    application.to_csv(path+'application_all.csv')
    application=pd.read_csv(path+'application_all.csv', index_col=0)
else :
    application=pd.read_csv(path+'application_all.csv', index_col=0)

##################### Modelisation ################################""

feat_eng3=False
if not feat_eng3:
    B2=bureau_avec_features.iloc[:,-6:-4].join(bureau_avec_features.iloc[:,-3:]).join(bureau_avec_features.iloc[:,0])
    B2=B2.drop_duplicates()
    application_final=pd.merge(application, B2, on='SK_ID_CURR')
    application_final.to_csv(path+'application_final.csv')
    application_final=pd.read_csv(path+'application_final.csv', index_col=0)

else :
    application_final=pd.read_csv(path+'application_final.csv', index_col=0)


##################### classification des variables ################################""
test_set=application_final[application_final['TARGET'].isnull()]
train_set=application_final[application_final['TARGET'].notnull()]

label=train_set['TARGET'] #label

features=application_final.columns[\
    np.isin(application_final.columns,'TARGET', invert=True)] #features

col_cat=application_final.loc[:,features]\
    .select_dtypes(exclude='number').columns #colonnes de catégories
#colonnes_numeriques
col_oth=application_final.loc[:,features]\
    .select_dtypes(include='number').columns #colonnes non catégorielles

col_boo=col_oth[col_oth.str.contains('FLAG')\
    |col_oth.str.contains('REGION_NOT')|\
        col_oth.str.contains('CITY_NOT')] #colonnes de booleens

col_num=col_oth[np.isin(col_oth, col_boo, invert=True)] #colonnes_numeriques

###########imputation des valeurs manquantes

def imputation(df,col_num, col_cat, col_boo):
    sp1=SimpleImputer(strategy='mean')
    col_num2=pd.DataFrame(sp1.fit_transform(df.loc[:,col_num]))
    col_num2.columns=col_num
    col_num2

    sp2=SimpleImputer(strategy='constant', fill_value='inconnu')
    col_cat2=pd.DataFrame(sp2.fit_transform(df.loc\
        [:,pd.concat([df.loc[:,col_cat]\
            ,df.loc[:,col_boo]], axis=1).columns]))
    col_cat2.columns=col_cat
    col_cat2
    return col_num2, col_cat2

col_num2, col_cat2=imputation(train_set,col_num, col_cat, col_boo)

sp1=SimpleImputer(strategy='mean')
col_num2=pd.DataFrame(sp1.fit_transform(train_set.loc[:,col_num]))
col_num2.columns=col_num
col_num2

sp2=SimpleImputer(strategy='constant', fill_value='inconnu')
col_cat2=pd.DataFrame(sp2.fit_transform(train_set.loc\
    [:,pd.concat([train_set.loc[:,col_cat]\
        ,train_set.loc[:,col_boo]], axis=1).columns]))
col_cat2.columns=col_cat
col_cat2

#pd.concat([application_final.loc[:,col_cat],application_final.loc[:,col_boo]], axis=1).columns

##################" preprocessing ##############################"

ohe=OneHotEncoder(sparse=False)
col_cat3=pd.DataFrame(ohe.fit_transform(col_cat2))

ss=StandardScaler()
col_num3=pd.DataFrame(ss.fit_transform(col_num2))

all_feat_ok=pd.concat([col_cat3,col_num3], axis=1)
all_feat_ok

#######################" modelisation ############################"
lr=LogisticRegression(max_iter=1000)

lr.fit(all_feat_ok, label)
lr

