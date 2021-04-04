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
    ,roc_auc_score, ConfusionMatrixDisplay, plot_roc_curve\
        , plot_confusion_matrix, plot_precision_recall_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

pd.set_option('display.max_colwidth', 40)
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

########################### feature engeneering #####################
deja_fait=True
if not deja_fait:
    fc.feature_engineering(path,False,False,False)

##################### classification des variables ################################""
application_final=pd.read_csv(path+'application_final.csv', index_col=0)

#séparation du train set et du test set, reuperation des noms de colonnes
train_set,test_set,label,col_cat,col_num,features = fc.preparation_df(application_final)

#### train test split du train set #######################
X_train, X_test, y_train, y_test = train_test_split(train_set[features],\
     train_set['TARGET'], test_size=0.25, random_state=0, stratify=train_set['TARGET'])

####### imputation des valeurs manquantes
col_num_train, col_cat_train=fc.imputation_valeurs_manquantes(X_train,col_num, col_cat)
col_num_test, col_cat_test=fc.imputation_valeurs_manquantes(X_test,col_num, col_cat)
#_, ohe_all,_=preprocessing(col_num_all,col_cat_all)

##################" preprocessing ##############################"
#libellé des colonnes catégories
cat_col_cat = [col_cat_train[column].unique() for column in col_cat_train]
df_final_train, ohe_train,ss_train=fc.preprocessing(col_num_train,col_cat_train, cat_col_cat)
df_final_test,ohe_test,ss_test=fc.preprocessing(col_num_test,col_cat_test,cat_col_cat)

################Récupère le nom des colonnes du df train final#######""
tab_nom_col_cat=[]
tab_nom_col=[]
for i in col_cat_train.columns:
    for j in col_cat_train[i].unique():
        tab_nom_col_cat.append('{}_{}'.format(i,j))
        tab_nom_col.append('{}_{}'.format(i,j))
for i in col_num_train.columns:
    tab_nom_col.append(i)

#####################gestion du déséquilibre du dataset###########
#meth=['SMOTE', 'RandomUnderSampler','Class_weight','Aucune']
#hyperp=[0.01,0.1,1,10]
meth=['SMOTE', 'RandomUnderSampler','Class_weight','Aucune']
hyperp=[0.01,0.1,1,10]
df_resultats,_=fc.modelisation(df_final_train,y_train,df_final_test,y_test,meth, hyperp)

df_resultats.to_csv(path+'df_resultats.csv')
df_resultats['Taux_Faux_Negatifs']=df_resultats['False Negative']/df_resultats.iloc[:,-4:].sum(axis=1)
df_resultats.sort_values('AUC',ascending=False)

############## modelisation avec les meilleurs paramètres ######################
meth2=['SMOTE']
hyperp2=[0.01]
df_resultats2,coefs=fc.modelisation(df_final_train,y_train,df_final_test,y_test,meth2, hyperp2)

############### étude des coefficients ###########
df_coef=pd.concat([pd.Series(tab_nom_col),pd.Series(lr.coef_[0])],axis=1)
df_coef.columns=['Nom_colonne','Coef']
df_coef['AbsCoef']=np.abs(df_coef['Coef'])
df_coef=df_coef.sort_values('AbsCoef',ascending=False)
df_coef.tail(20)
df_coef.to_csv(path+'coefficients.csv')

##### Evaluation (scores et graphs)
fc.scores(y_test,resu_lr,lr,df_final_test)
