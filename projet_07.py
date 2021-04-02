import pandas as pd
from sklearn import impute
import sklearn
import fonctions as fc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,auc,f1_score,roc_auc_score, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV

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

########################### feature engeneering #####################

########1ere partie##########

feat_eng1=True
if not feat_eng1:
    bureau=pd.read_csv(path+'bureau.csv')
    B=fc.create_bureau_features(bureau) #creation de 10 features
    B.to_csv(path+'bureau_avec_features.csv')
    print("fini")
    bureau_avec_features=pd.read_csv(path+'bureau_avec_features.csv', index_col=0)
else :
    bureau_avec_features=pd.read_csv(path+'bureau_avec_features.csv', index_col=0)
1+2
########## 2eme partie ########

feat_eng2=True
if not feat_eng2:
    application_test=pd.read_csv(path+'application_test.csv')
    application_train=pd.read_csv(path+'application_train.csv')
    application=fc.feat_eng(application_train, application_test)
    application.to_csv(path+'application_all.csv')
    application=pd.read_csv(path+'application_all.csv', index_col=0)
else :
    application=pd.read_csv(path+'application_all.csv', index_col=0)

########### 3eme partie #######

feat_eng3=False
if not feat_eng3:
    B2=bureau_avec_features.iloc[:,-7:-5].join(bureau_avec_features.iloc[:,-4:]).join(bureau_avec_features.iloc[:,0])
    B2=B2.drop_duplicates()
    del bureau_avec_features
    print('ok1')
    application_final=pd.merge(application, B2, on='SK_ID_CURR')
    application_final.to_csv(path+'application_final.csv')
    application_final=pd.read_csv(path+'application_final.csv', index_col=0)
    print('ok2')

##################### classification des variables ################################""
application_final=pd.read_csv(path+'application_final.csv', index_col=0)

test_set=application_final[application_final['TARGET'].isnull()]
train_set=application_final[application_final['TARGET'].notnull()]
label=train_set['TARGET'] #label

features=application_final.columns[\
    np.isin(application_final.columns,'TARGET', invert=True)] #features
col_not_num=application_final.loc[:,features]\
    .select_dtypes(exclude='number').columns #colonnes non numériques
col_oth=application_final.loc[:,features]\
    .select_dtypes(include='number').columns #autres colonnes
col_boo=col_oth[col_oth.str.contains('FLAG')\
    |col_oth.str.contains('REGION_NOT')|\
        col_oth.str.contains('CITY_NOT')] #colonnes de booleens parmi les autres colonnes
col_num=col_oth[np.isin(col_oth, col_boo, invert=True)] #colonnes_numeriques
col_cat=features[np.isin(features, col_num, invert=True)] #colonnes_categorielles
del col_not_num, col_oth


#application_final.loc[:,col_cat]=application_final.loc[:,col_cat].fillna('Inconnu')


for i in col_cat:
    application_final.loc[:,i]=application_final.loc[:,i].astype('str')

cat_col_cat = [application_final[column].unique() for column in application_final[col_cat]]
cat_col_cat


#for i in application_final[col_cat] :
#    print(application_final[i].value_counts())






#### train test split #######################
X_train, X_test, y_train, y_test = train_test_split(train_set[features],\
     train_set['TARGET'], test_size=0.25, random_state=0, stratify=train_set['TARGET'])

### imputation des valeurs manquantes

def imputation_valeurs_manquantes(df,col_num, col_cat):
    sp1=SimpleImputer(strategy='mean')
    col_num2=pd.DataFrame(sp1.fit_transform(df.loc[:,col_num]))
    col_num2.columns=col_num

    sp2=SimpleImputer(strategy='constant', fill_value='inconnu')
    col_cat2=pd.DataFrame(sp2.fit_transform(df.loc[:,col_cat]))
    col_cat2.columns=col_cat
    return col_num2, col_cat2

col_num_train, col_cat_train=imputation_valeurs_manquantes(X_train,col_num, col_cat)
col_num_test, col_cat_test=imputation_valeurs_manquantes(X_test,col_num, col_cat)


##################" preprocessing ##############################"
def preprocessing(col_num,col_cat):
    ohe=OneHotEncoder(sparse=False, handle_unknown='ignore', categories=cat_col_cat)
    col_cat_tr=pd.DataFrame(ohe.fit_transform(col_cat))

    ss=StandardScaler()
    col_num_tr=pd.DataFrame(ss.fit_transform(col_num))

    all_feat_ok=pd.concat([col_cat_tr,col_num_tr], axis=1)
    return all_feat_ok

df_final_train=preprocessing(col_num_train,col_cat_train)
df_final_test=preprocessing(col_num_test,col_cat_test)

df_final_test.shape
df_final_train.shape

#######################" modelisation ############################"

lr=LogisticRegression(max_iter=2000)
param={'C':[0.1,1,10]}
gs=GridSearchCV(lr,param_grid=param, scoring='f1')

gs.fit(df_final_train, y_train)

resu=gs.predict(df_final_test)
resu

accuracy_score(y_test,resu)
confusion_matrix(y_test,resu)
cm=ConfusionMatrixDisplay(confusion_matrix(y_test,resu))
cm.plot()
plt.show()
roc_auc_score(y_test,resu)
f1_score(y_test,resu)
