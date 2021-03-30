import pandas as pd
import fonctions as fc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_colwidth', 40)
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

if not feat_eng1:
    application_test=pd.read_csv(path+'application_test.csv')
    application_train=pd.read_csv(path+'application_train.csv')
    df=fc.feat_eng(application_train, application_test)
    df.to_csv(path+'application_all.csv')
else :
    application=pd.read_csv(path+'application_all.csv')

##################################################################
bureau_avec_features.head()
bureau_avec_features.columns
bureau_avec_features.iloc[0]

info_colonnes
credit_card_balance

##################### Modelisation ################################""