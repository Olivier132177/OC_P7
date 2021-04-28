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
dict_name_income_type={
'Working':'Working', 
'State servant':'State servant', 
'Commercial associate':'other',
'Pensioner':'Pensioner',
'Unemployed':'other', 
'Student':'other', 
'Businessman':'other', 
'Maternity leave':'other'}

dict_occupation_type={
'Laborers':'Laborers', 
'Core staff':'Core staff', 
'Accountants':'Accountants', 
'Managers':'Managers', 
'Drivers':'Drivers', 
'Sales staff':'Sales staff', 
'Cleaning staff':'Cleaning staff', 
'Cooking staff':'Cooking staff',
'Private service staff':'Private service staff', 
'Medicine staff':'Medicine staff', 
'Security staff':'Security staff',
'Waiters/barmen staff':'Waiters/barmen staff', 
'Low-skill Laborers':'other', 
'Realty agents':'other',
'Secretaries':'Secretaries', 
'High skill tech staff':'High skill tech staff', 
'IT staff':'IT staff', 
'HR staff':'other'}

dict_organization_type={
'Business Entity Type 3':'Business Entity Type 3',
'School':'School', 
'Government':'Government', 
'Religion':'other',
'Other':'other',
'XNA':'XNA', 
'Medicine':'Medicine', 
'Business Entity Type 2':'Business Entity Type 2',
'Self-employed':'Self-employed', 
'Housing':'other', 
'Kindergarten':'Kindergarten', 
'Trade: type 7':'Trade: type 7',
'Industry: type 11':'other', 
'Military':'Military', 
'Services':'Services', 
'Transport: type 4':'Transport: type 4',
'Industry: type 1':'Industry: type 1', 
'Emergency':'other', 
'Security':'other', 
'Trade: type 2':'Trade: type 2',
'University':'University', 
'Transport: type 3':'Transport: type 3', 
'Police':'Police', 
'Construction':'Construction',
'Business Entity Type 1':'other', 
'Postal':'Postal', 
'Industry: type 4':'other',
'Agriculture':'Agriculture', 
'Restaurant':'Restaurant', 
'Transport: type 2':'other', 
'Culture':'other',
'Hotel':'Hotel', 
'Industry: type 7':'Industry: type 7', 
'Trade: type 3':'Trade: type 3', 
'Industry: type 3':'Industry: type 3',
'Bank':'Bank', 
'Industry: type 9':'Industry: type 9', 
'Trade: type 6':'Trade: type 6', 
'Industry: type 2':'Industry: type 2',
'Transport: type 1':'other', 
'Electricity':'Electricity', 
'Industry: type 12':'Industry: type 12',
'Insurance':'other', 
'Security Ministries':'Security Ministries', 
'Mobile':'Mobile', 
'Trade: type 1':'other',
'Industry: type 5':'other', 
'Industry: type 10':'other', 
'Legal Services':'Legal Services',
'Advertising':'Advertising', 
'Trade: type 5':'other', 
'Cleaning':'Cleaning', 
'Industry: type 13':'other',
'Industry: type 8':'other', 
'Realtor':'Realtor', 
'Telecom':'other', 
'Industry: type 6':'other',
'Trade: type 4':'other'}

application_final['LN_REVENU_TOTAL']=np.log(application_final['AMT_INCOME_TOTAL'])
application_final['NAME_INCOME_TYPE']=application_final['NAME_INCOME_TYPE'].map(dict_name_income_type)
application_final['OCCUPATION_TYPE']=application_final['OCCUPATION_TYPE'].map(dict_occupation_type)
application_final['ORGANIZATION_TYPE']=application_final['ORGANIZATION_TYPE'].map(dict_organization_type)

application_final['OCCUPATION_TYPE'].unique()
application_final['NAME_INCOME_TYPE'].unique()
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
            'EXT_SOURCE_1_y','EXT_SOURCE_2_y','EXT_SOURCE_3_y','DAYS_BIRTH_y',
            'FLAG_DOCUMENT_2','AMT_REQ_CREDIT_BUREAU_HOUR','FLAG_DOCUMENT_12','FLAG_DOCUMENT_4',
            'FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_5','WALLSMATERIAL_MODE',
            'FLAG_DOCUMENT_19','EXT_SOURCES_MEAN']

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

nom_colonnes={'NAME_CONTRACT_TYPE':'TYPE_CONTRAT',
              'CODE_GENDER':'SEXE', 
              'FLAG_OWN_CAR' :'POSSEDE_UNE_VOITURE',
              'AMT_INCOME_TOTAL':'REVENUS_CLIENTS',
              'AMT_CREDIT':'MONTANT_CREDIT',
              'AMT_ANNUITY':'MONTANT ANNUITE',
              'AMT_GOODS_PRICE':'PRIX_DU_BIEN',
              'NAME_TYPE_SUITE': 'QUI ACCOMPAGNE LE CLIENT?',
              'NAME_INCOME_TYPE':'TYPE_DE_REVENUS', 
              'NAME_EDUCATION_TYPE':'NIVEAU_D_ETUDES',
              'NAME_FAMILY_STATUS':'STATUT FAMILIAL',
              'NAME_HOUSING_TYPE':'LOGEMENT_ACTUEL',
              'REGION_POPULATION_RELATIVE':'POPULATION_DE_LA_REGION_NORMALISEE',
              'DAYS_BIRTH_x':'AGE_EN_JOURS', 
              'DAYS_EMPLOYED':'EMPLOYE DEPUIS', 
              'DAYS_REGISTRATION' :'NOMBRE_DE_JOURS_DEPUIS_MISE_A_JOUR_PAPIERS', 
              'DAYS_ID_PUBLISH':'NOMBRE_DE_JOURS_DEPUIS_CHANGEMENT_PAPIERS_IDENTITE',
              'FLAG_WORK_PHONE':'A_FOURNI_NUM_TELEPHONE_TRAVAIL',
              'EXT_SOURCE_2_x':'SCORE_SOURCE_1_NORMALISE',
              'OCCUPATION_TYPE':'PROFESSION', 
              'REGION_RATING_CLIENT':'NOTE_REGION',
              'REGION_RATING_CLIENT_W_CITY':'NOTE_REGION_AVEC_PRISE_EN_COMPTE_DE_LA_VILLE',
              'LIVE_REGION_NOT_WORK_REGION':'ADRESSE_CONTACT_CORRESPOND_A_L_ADRESSE_TRAVAIL',
              'REG_CITY_NOT_LIVE_CITY':'ADRESSE_CONTACT_DIFFERENTE_DE_L_ADRESSE_PERMANENTE',
              'LIVE_CITY_NOT_WORK_CITY':'ADRESSE_TRAVAIL_DIFFERENTE_DE_L_ADRESSE_PERMANENTE',
              'ORGANIZATION_TYPE':'TYPE_SOCIETE',
              'EXT_SOURCE_1_x':'SCORE_SOURCE_EXTERIEURE_1_NORMALISE',
              'EXT_SOURCE_2_x':'SCORE_SOURCE_EXTERIEURE_2_NORMALISE',
              'EXT_SOURCE_3_x':'SCORE_SOURCE_EXTERIEURE_3_NORMALISE',
              'DEF_30_CNT_SOCIAL_CIRCLE':'NOMBRE_DEFAUTS_30_DPD_OBSERVES_DANS_CERCLE_SOCIAL',
              'DEF_60_CNT_SOCIAL_CIRCLE':'NOMBRE_DEFAUTS_60_DPD_OBSERVES_DANS_CERCLE_SOCIAL', 
              'DAYS_LAST_PHONE_CHANGE':'NOMBRE_JOURS_DEPUIS_CHANGEMENT_NUM_TELEPHONE', 
              'FLAG_DOCUMENT_3':'A_FOURNI_LE_DOCUMENT_3',
              'FLAG_DOCUMENT_4': 'A_FOURNI_LE_DOCUMENT_4',
              'FLAG_DOCUMENT_5':'A_FOURNI_LE_DOCUMENT_5', 
              'FLAG_DOCUMENT_6':'A_FOURNI_LE_DOCUMENT_6',
              'FLAG_DOCUMENT_7':'A_FOURNI_LE_DOCUMENT_7', 
              'FLAG_DOCUMENT_8':'A_FOURNI_LE_DOCUMENT_8', 
              'FLAG_DOCUMENT_9':'A_FOURNI_LE_DOCUMENT_9',
              'FLAG_DOCUMENT_10':'A_FOURNI_LE_DOCUMENT_10', 
              'FLAG_DOCUMENT_11':'A_FOURNI_LE_DOCUMENT_11', 
              'FLAG_DOCUMENT_12':'A_FOURNI_LE_DOCUMENT_12',
              'FLAG_DOCUMENT_13':'A_FOURNI_LE_DOCUMENT_13', 
              'FLAG_DOCUMENT_14':'A_FOURNI_LE_DOCUMENT_14', 
              'FLAG_DOCUMENT_15':'A_FOURNI_LE_DOCUMENT_15',
              'FLAG_DOCUMENT_16':'A_FOURNI_LE_DOCUMENT_16', 
              'FLAG_DOCUMENT_17':'A_FOURNI_LE_DOCUMENT_17', 
              'FLAG_DOCUMENT_18':'A_FOURNI_LE_DOCUMENT_18',
              'FLAG_DOCUMENT_19':'A_FOURNI_LE_DOCUMENT_19', 
              'FLAG_DOCUMENT_20':'A_FOURNI_LE_DOCUMENT_20', 
              'FLAG_DOCUMENT_21':'A_FOURNI_LE_DOCUMENT_21',
              'AMT_REQ_CREDIT_BUREAU_QRT':'NOMBRE_DEMANDES_CLIENTS_3_MOIS_AUPARAVANT', 
              'family_members_more7':'family_members_more7',
              'islowskilled_labour':'islowskilled_labour', 
              'is_Maternity_leave':'is_Maternity_leave', 
              'is_unemployed':'is_unemployed',
              'cnt_childern_more6':'cnt_childern_more6', 
              'CREDIT_INCOME_PERCENT':'CREDIT_INCOME_PERCENT', 
              'ANNUITY_INCOME_PERCENT':'ANNUITY_INCOME_PERCENT',
              'CREDIT_TERM':'CREDIT_TERM', 
              'DAYS_EMPLOYED_PERCENT':'DAYS_EMPLOYED_PERCENT', 
              'CREDIT_TO_GOODS_RATIO':'CREDIT_TO_GOODS_RATIO',
              'INCOME_TO_EMPLOYED_RATIO':'INCOME_TO_EMPLOYED_RATIO', 
              'INCOME_TO_BIRTH_RATIO':'INCOME_TO_BIRTH_RATIO',
              'ID_TO_BIRTH_RATIO':'ID_TO_BIRTH_RATIO', 
              'PHONE_TO_BIRTH_RATIO':'PHONE_TO_BIRTH_RATIO', 
              'EXT_SOURCES_MIN':'EXT_SOURCES_MIN',
              'EXT_SOURCES_MAX':'EXT_SOURCES_MAX', 
              'EXT_SOURCES_MEAN':'EXT_SOURCES_MEAN', 
              'EXT_SOURCES_NANMEDIAN':'EXT_SOURCES_NANMEDIAN',
              'EXT_SOURCES_VAR':'EXT_SOURCES_VAR', 
              'CREDIT_LENGTH':'DUREE_DU_CREDIT', 
              'AVERAGE_LOAN_TYPE':'AVERAGE_LOAN_TYPE',
              'ACTIVE_LOANS_PERCENTAGE':'ACTIVE_LOANS_PERCENTAGE', 
              'CREDIT_ENDDATE_PERCENTAGE':'CREDIT_ENDDATE_PERCENTAGE',
              'AVG_ENDDATE_FUTURE':'AVG_ENDDATE_FUTURE', 
              'DEBT_CREDIT_RATIO':'DEBT_CREDIT_RATIO', 
              'OVERDUE_DEBT_RATIO':'OVERDUE_DEBT_RATIO',
              'AVG_CREDITDAYS_PROLONGED':'AVG_CREDITDAYS_PROLONGED', 
              'LN_REVENU_TOTAL':'LN DES REVENUS CLIENT',
              'TARGET':'TARGET'}

application_final=application_final.rename(columns=nom_colonnes)

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
    algo=['LR'] 
    paramc=[0.01,0.1,1,10]
    parammd=[7,10,13,16]
    #meth=['Aucune','Class_weight','SMOTE', 'RandomUnderSampler']
    #algo=['RF','LR'] 
    #paramc=[0.01,0.1,1,10]
    #parammd=[7,8,9,10,11,12,13]
    df_resultats_2=fc.modelisation2(df_final_train,y_train,df_final_test,y_test,meth,algo,paramc,parammd,False)
   
    df_resultats_2['F_beta']=(5*df_resultats_2['Precision']*df_resultats_2['Recall'])\
    /((4*df_resultats_2['Precision'])+df_resultats_2['Recall'])
    df_resultats_2.to_csv(path+'df_resultats_apres.csv')

resultats_finaux=pd.read_csv(path+'df_resultatsF1B.csv')
resultats_finaux
############## modelisation avec des paramètres sélectionnés ######################
lr2=LogisticRegression(max_iter=2000, class_weight='balanced',random_state=0, C=0.05,penalty='l1',solver='saga')
lr2.fit(df_final_train, y_train)
y_pred=lr2.predict(df_final_test)
y_prob=lr2.predict_proba(df_final_test).T[1]

acc, mat, a_u_c, f1, auc_pr=fc.scores(y_test,y_pred,y_prob,lr2,df_final_test, graphs=True)

dump(lr2, 'modele_sauvegarde.joblib') 

df_final_test_pred=X_test.copy()
#df_final_test_pred['y_pred']=y_prob
df_final_test_pred.to_csv(path+'df_test_avant_transfo_pour_dashboard.csv') #pour les graphs/stats

df_final_test.index=X_test.index

df_final_test.to_csv(path+'df_test_prep_pour_dashboard.csv') #pour la modélisation


df_train_pour_dash=X_train.copy()
df_train_pour_dash['label']=y_train
df_train_pour_dash.to_csv(path+'df_train_pour_dashboard.csv') #pour les graphs/stats


coefs=lr2.coef_
df_coef=pd.concat([pd.Series(tab_nom_col),pd.Series(coefs[0])],axis=1)
df_coef.columns=['Variables','Coef']
df_coef['AbsCoef']=np.abs(df_coef['Coef'])
df_coef=df_coef.set_index('Variables')
df_coef.sort_values('AbsCoef').iloc[0:20]
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


########## Meilleurs résultats obtenus ############
#Méthode : Class_weight C : 0.01
#Matrice de confusion :[[42254 18526]
# [ 1567  3525]]
#Accuracy : 0.695 ROC AUC : 0.76 AUC Precision-Recall : 0.241 F1 : 0.26

########Derniere tentative
#Matrice de confusion :[[42247 18533]
# [ 1570  3522]]
#Accuracy : 0.695 ROC AUC : 0.76 AUC Precision-Recall : 0.24 F1 : 0.259

df_resultats_MN=pd.read_csv(path+'df_resultatsMN.csv', index_col=0)
df_resultats_MN.sort_values(['algo','methode'])

df_resultats_MN['numer_F1']=df_resultats_MN['Recall']*df_resultats_MN['Precision']*2
df_resultats_MN['denom_F1']=df_resultats_MN['Recall']+df_resultats_MN['Precision']

df_resultats_MN[['algo', 'methode', 'Best_params',  'Confusion_matrix',
       'Precision_Recall_AUC', 'Recall',
       'Precision', 'numer_F1','denom_F1','F1_score']].sort_values('F1_score')


df_resultats_MN.sort_values(['F_beta'])

application_final['NAME_INCOME_TYPE'].unique()



application_final.columns