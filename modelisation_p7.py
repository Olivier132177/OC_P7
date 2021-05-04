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
path2='/home/olivier/Desktop/openclassrooms/P7/data_dashboard/'

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


graph_imbala=False
if graph_imbala:
    imbala=application_train['TARGET'].value_counts(normalize=True)
    imbala=round(imbala*100,1)
    plt.bar(imbala.index,imbala.values)
    plt.xticks(imbala.index)
    plt.xlabel('Labels')
    plt.ylabel('Pourcentage du dataset (%)')
    plt.yticks(np.arange(0,105,5))
    plt.show(block=False)


#feature engeneering
deja_fait=True
if not deja_fait:
    fc.feature_engineering(path,False,False,False)
print('tout fini')
##################### suppression des variables inutiles ################################""
application_final=pd.read_csv(path+'application_final.csv', index_col=0).set_index('SK_ID_CURR')

application_final=fc.post_feat_eng(application_final) #renome les colonnes, fusionne des catégories

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
    meth=['Aucune','Class_weight','SMOTE', 'RandomUnderSampler']
    param_lr={'C':[0.01,0.03,0.1,0.3,1],'penalty':['l1','l2']}
    #parammd=[7,10,13,16]
    #meth=['Aucune','Class_weight','SMOTE', 'RandomUnderSampler']
    #algo=['RF','LR'] 
    #paramc=[0.01,0.1,1,10]
    #parammd=[7,8,9,10,11,12,13]
    df_resultats_2=fc.modelisation2(df_final_train,y_train,df_final_test,y_test,meth,param_lr,False)
   
    df_resultats_2['F_beta']=(5*df_resultats_2['Precision']*df_resultats_2['Recall'])\
    /((4*df_resultats_2['Precision'])+df_resultats_2['Recall'])
    df_resultats_2.to_csv(path+'df_resultats_apres5.csv')

resultats_finaux3=pd.read_csv(path+'df_resultats_apres3.csv', index_col=0)

resultats_finaux3

#rus = RandomUnderSampler(random_state=0)
#df_final_train_v2, y_train_v2 = rus.fit_resample\
#(np.array(df_final_train), np.array(y_train))

lr2=LogisticRegression(max_iter=2000,random_state=0, C=0.01, class_weight='balanced')#5,penalty='l1',solver='saga')
lr2.fit(df_final_train, y_train)
y_pred=lr2.predict(df_final_test)
y_prob=lr2.predict_proba(df_final_test).T[1]

acc, mat, a_u_c, f1, auc_pr=fc.scores(y_test,y_pred,y_prob,lr2,df_final_test, graphs=True)

meil_resultat=pd.Series({'Accuracy':acc,'ROC_AUC':a_u_c,'Precision_Recall_AUC' : auc_pr,'F1_score':f1,
        'Confusion_matrix':mat,'True Negative':mat[0,0],'True Positive':mat[1,1],
        'False Positive': mat[0,1],'False Negative': mat[1,0]})
meil_resultat['Precision']=meil_resultat['True Positive']/(meil_resultat['True Positive']+meil_resultat['False Positive'])
meil_resultat['Recall']=meil_resultat['True Positive']/(meil_resultat['True Positive']+meil_resultat['False Negative'])

coef_beta=2
intitu='F_beta_score_{}'.format(coef_beta)
meil_resultat[intitu]=((1+coef_beta**2)*meil_resultat['Precision']*meil_resultat['Recall'])\
    /(((coef_beta**2)*meil_resultat['Precision'])+meil_resultat['Recall'])

meil_resultat

dump(lr2, 'modele_sauvegarde.joblib') 

df_final_test_pred=X_test.copy()
#df_final_test_pred['y_pred']=y_prob
df_final_test_pred.head(100).to_csv('df_test_avant_transfo_pour_dashboard.csv') #pour les graphs/stats

df_final_test.index=X_test.index

df_final_test.head(100).to_csv('df_test_prep_pour_dashboard.csv') #pour la modélisation

df_train_pour_dash=X_train.copy()
df_train_pour_dash['label']=y_train
df_train_pour_dash.head(20000).to_csv('df_train_pour_dashboard.csv') #pour les graphs/stats

coefs=lr2.coef_

df_coef=pd.concat([pd.Series(tab_nom_col),pd.Series(coefs[0])],axis=1)
df_coef.columns=['Variables','Coef']
df_coef['AbsCoef']=np.abs(df_coef['Coef'])
df_coef=df_coef.set_index('Variables')
df_coef.to_csv('coefficients.csv')

###### #### Meilleurs résultats obtenus #### #### ####
#Méthode : Class_weight C : 0.01
#Matrice de confusion :[[42257 18523]
# [ 1567  3525]]
#Accuracy : 0.695 ROC AUC : 0.759 AUC Precision - Recall : 0.24 F1 : 0.26
