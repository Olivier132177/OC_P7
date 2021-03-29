import pandas as pd
import fonctions as fc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

pd.set_option('display.max_colwidth', 40)

application_test=pd.read_csv('application_test.csv')
application_train=pd.read_csv('application_train.csv')
bureau_balance=pd.read_csv('bureau_balance.csv')
bureau=pd.read_csv('bureau.csv')
credit_card_balance=pd.read_csv('credit_card_balance.csv')
info_colonnes=pd.read_csv('HomeCredit_columns_description.csv', encoding='cp850')
installments_payments=pd.read_csv('installments_payments.csv')
pos_cash_balance=pd.read_csv('POS_CASH_balance.csv')
previous_application=pd.read_csv('previous_application.csv')
sample_submission=pd.read_csv('sample_submission.csv')

########################### feature engeneering 1 #####################

feat_eng1=True
if not feat_eng1:
    B=fc.create_bureau_features(bureau) #creation de 10 features
    B.to_csv('bureau_avec_features.csv')
    print("fini")
else :
    bureau_avec_features=pd.read_csv('bureau_avec_features.csv')

########################### feature engeneering 2 #####################

df=fc.feat_eng(application_train, application_test)

#comment


