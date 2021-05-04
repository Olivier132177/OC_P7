import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix,accuracy_score,auc,f1_score\
    ,roc_auc_score, precision_recall_curve, plot_roc_curve\
        , plot_confusion_matrix, plot_precision_recall_curve,auc\
            ,fbeta_score, make_scorer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def create_bureau_features(bureau): # de HOME    CREDIT - BUREAU DATA - FEATURE ENGINEERING sur Kaggle
#1ere partie du feature engeneering
    #FEATURE 1 - NUMBER OF PAST LOANS PER CUSTOMER
    B = bureau
    grp = B[['SK_ID_CURR', 'DAYS_CREDIT']].groupby(by = ['SK_ID_CURR'])['DAYS_CREDIT'].count().reset_index().rename(index=str, columns={'DAYS_CREDIT': 'BUREAU_LOAN_COUNT'})
    B = B.merge(grp, on = ['SK_ID_CURR'], how = 'left')
    print('feature 1 ok')

    #FEATURE 2 - NUMBER OF TYPES OF PAST LOANS PER CUSTOMER
    grp = B[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(by = ['SK_ID_CURR'])['CREDIT_TYPE'].nunique().reset_index().rename(index=str, columns={'CREDIT_TYPE': 'BUREAU_LOAN_TYPES'})
    B = B.merge(grp, on = ['SK_ID_CURR'], how = 'left')
    print('feature 2 ok')

    #FEATURE 3 - AVERAGE NUMBER OF PAST LOANS PER TYPE PER CUSTOMER
    #Is the Customer diversified in taking multiple types of Loan or Focused on a single type of loan

    # Number of Loans per Customer
    grp = B[['SK_ID_CURR', 'DAYS_CREDIT']].groupby(by = ['SK_ID_CURR'])['DAYS_CREDIT'].count().reset_index().rename(index=str, columns={'DAYS_CREDIT': 'BUREAU_LOAN_COUNT'})
    B = B.merge(grp, on = ['SK_ID_CURR'], how = 'left')

    # Number of types of Credit loans for each Customer 
    grp = B[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(by = ['SK_ID_CURR'])['CREDIT_TYPE'].nunique().reset_index().rename(index=str, columns={'CREDIT_TYPE': 'BUREAU_LOAN_TYPES'})
    B = B.merge(grp, on = ['SK_ID_CURR'], how = 'left')

    # Average Number of Loans per Loan Type
    B=B.iloc[:,:-2]
    B['AVERAGE_LOAN_TYPE'] = B['BUREAU_LOAN_COUNT_x']/B['BUREAU_LOAN_TYPES_x']
    del B['BUREAU_LOAN_COUNT_x'], B['BUREAU_LOAN_TYPES_x']
    print('feature 3 ok')

    #FEATURE 4 - % OF ACTIVE LOANS FROM BUREAU DATA
    B['CREDIT_ACTIVE_BINARY'] = B['CREDIT_ACTIVE']

    def f(x):
        if x == 'Closed':
            y = 0
        else:
            y = 1    
        return y

    # Create a new dummy column for whether CREDIT is ACTIVE OR CLOED 
    B['CREDIT_ACTIVE_BINARY'] = B.apply(lambda x: f(x.CREDIT_ACTIVE), axis = 1)
    grp = B.groupby(by = ['SK_ID_CURR'])['CREDIT_ACTIVE_BINARY'].mean().reset_index().rename(index=str, columns={'CREDIT_ACTIVE_BINARY': 'ACTIVE_LOANS_PERCENTAGE'})
    # Calculate mean number of loans that are ACTIVE per CUSTOMER 
    B = B.merge(grp, on = ['SK_ID_CURR'], how = 'left')
    del B['CREDIT_ACTIVE_BINARY']

    print('feature 4 ok')
    #FEATURE 5
    #AVERAGE NUMBER OF DAYS BETWEEN SUCCESSIVE PAST APPLICATIONS FOR EACH CUSTOMER
    #How often did the customer take credit in the past? Was it spaced out at regular time intervals 
    # - a signal of good financial planning OR were the loans concentrated around a smaller time frame - indicating potential financial trouble?

    # Groupby each Customer and Sort values of DAYS_CREDIT in ascending order
    grp = B[['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT']].groupby(by = ['SK_ID_CURR'])
    grp1 = grp.apply(lambda x: x.sort_values(['DAYS_CREDIT'], ascending = False)).reset_index(drop = True)#rename(index = str, columns = {'DAYS_CREDIT': 'DAYS_CREDIT_DIFF'})
    print("Grouping and Sorting done")

    # Calculate Difference between the number of Days 
    grp1['DAYS_CREDIT1'] = grp1['DAYS_CREDIT']*-1
    grp1['DAYS_DIFF'] = grp1.groupby(by = ['SK_ID_CURR'])['DAYS_CREDIT1'].diff()
    grp1['DAYS_DIFF'] = grp1['DAYS_DIFF'].fillna(0).astype('uint32')
    del grp1['DAYS_CREDIT1'], grp1['DAYS_CREDIT'], grp1['SK_ID_CURR']
    print("Difference days calculated")

    B = B.merge(grp1, on = ['SK_ID_BUREAU'], how = 'left')
    print("Difference in Dates between Previous CB applications is CALCULATED ")
    print('feature 5 ok')

    #FEATURE 6 % of LOANS PER CUSTOMER WHERE END DATE FOR CREDIT IS PAST
    #INTERPRETING CREDIT_DAYS_ENDDATE
    #NEGATIVE VALUE - Credit date was in the past at time of application( Potential Red Flag !!! )
    #POSITIVE VALUE - Credit date is in the future at time of application ( Potential Good Sign !!!!)
    #NOTE : This is not the same as % of Active loans since Active loans
    #can have Negative and Positive values for DAYS_CREDIT_ENDDATE¶

    B['CREDIT_ENDDATE_BINARY'] = B['DAYS_CREDIT_ENDDATE']

    def f(x):
        if x<0:
            y = 0
        else:
            y = 1   
        return y

    B['CREDIT_ENDDATE_BINARY'] = B.apply(lambda x: f(x.DAYS_CREDIT_ENDDATE), axis = 1)
    print("New Binary Column calculated")

    grp = B.groupby(by = ['SK_ID_CURR'])['CREDIT_ENDDATE_BINARY'].mean().reset_index().rename(index=str, columns={'CREDIT_ENDDATE_BINARY': 'CREDIT_ENDDATE_PERCENTAGE'})
    B = B.merge(grp, on = ['SK_ID_CURR'], how = 'left')

    del B['CREDIT_ENDDATE_BINARY']
    print('feature 6 ok')

    #FEATURE 7
    #AVERAGE NUMBER OF DAYS IN WHICH CREDIT EXPIRES IN FUTURE -INDICATION OF CUSTOMER DELINQUENCY IN FUTURE??

    # Repeating Feature 6 to Calculate all transactions with ENDATE as POSITIVE VALUES 

    # Dummy column to calculate 1 or 0 values. 1 for Positive CREDIT_ENDDATE and 0 for Negative
    B['CREDIT_ENDDATE_BINARY'] = B['DAYS_CREDIT_ENDDATE']

    def f(x):
        if x<0:
            y = 0
        else:
            y = 1   
        return y

    B['CREDIT_ENDDATE_BINARY'] = B.apply(lambda x: f(x.DAYS_CREDIT_ENDDATE), axis = 1)
    print("New Binary Column calculated")

    # We take only positive values of  ENDDATE since we are looking at Bureau Credit VALID IN FUTURE 
    # as of the date of the customer's loan application with Home Credit 
    B1 = B.copy()
    B1 = B1.loc[B1['CREDIT_ENDDATE_BINARY'] == 1]
    B1.shape

    #Calculate Difference in successive future end dates of CREDIT 

    # Create Dummy Column for CREDIT_ENDDATE 
    B1['DAYS_CREDIT_ENDDATE1'] = B1.loc[:,'DAYS_CREDIT_ENDDATE']
    # Groupby Each Customer ID 
    grp = B1[['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT_ENDDATE1']].groupby(by = ['SK_ID_CURR'])
    # Sort the values of CREDIT_ENDDATE for each customer ID 
    grp1 = grp.apply(lambda x: x.sort_values(['DAYS_CREDIT_ENDDATE1'], ascending = True)).reset_index(drop = True)
    del grp
    print("Grouping and Sorting done")

    # Calculate the Difference in ENDDATES and fill missing values with zero 
    grp1['DAYS_ENDDATE_DIFF'] = grp1.groupby(by = ['SK_ID_CURR'])['DAYS_CREDIT_ENDDATE1'].diff()
    grp1['DAYS_ENDDATE_DIFF'] = grp1['DAYS_ENDDATE_DIFF'].fillna(0).astype('uint32')
    del grp1['DAYS_CREDIT_ENDDATE1'], grp1['SK_ID_CURR']
    print("Difference days calculated")

    # Merge new feature 'DAYS_ENDDATE_DIFF' with original Data frame for BUREAU DATA
    B = B.merge(grp1, on = ['SK_ID_BUREAU'], how = 'left')
    del grp1

    # Calculate Average of DAYS_ENDDATE_DIFF

    grp = B[['SK_ID_CURR', 'DAYS_ENDDATE_DIFF']].groupby(by = ['SK_ID_CURR'])['DAYS_ENDDATE_DIFF'].mean().reset_index().rename( index = str, columns = {'DAYS_ENDDATE_DIFF': 'AVG_ENDDATE_FUTURE'})
    B = B.merge(grp, on = ['SK_ID_CURR'], how = 'left')
    del grp 
    del B['DAYS_ENDDATE_DIFF']
    del B['CREDIT_ENDDATE_BINARY'], B['DAYS_CREDIT_ENDDATE']
    
    # Verification of Feature 
    #B[B['SK_ID_CURR'] == 100653]
    # In the Data frame below we have 3 values not NAN 
    # Average of 3 values = (0 +0 + 3292)/3 = 1097.33 
    #The NAN Values are Not Considered since these values DO NOT HAVE A FUTURE CREDIT END DATE 
    print('feature 7 ok')

    #FEATURE 8 - DEBT OVER CREDIT RATIO
    #The Ratio of Total Debt to Total Credit for each Customer
    #A High value may be a red flag indicative of potential default

    B[~B['AMT_CREDIT_SUM_LIMIT'].isnull()][0:2]

    # WE can see in the Table Below 
    # AMT_CREDIT_SUM = AMT_CREDIT_SUM_DEBT + AMT_CREDIT_SUM_LIMIT

    B['AMT_CREDIT_SUM_DEBT'] = B['AMT_CREDIT_SUM_DEBT'].fillna(0)
    B['AMT_CREDIT_SUM'] = B['AMT_CREDIT_SUM'].fillna(0)

    grp1 = B[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT']].groupby(by = ['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename( index = str, columns = { 'AMT_CREDIT_SUM_DEBT': 'TOTAL_CUSTOMER_DEBT'})
    grp2 = B[['SK_ID_CURR', 'AMT_CREDIT_SUM']].groupby(by = ['SK_ID_CURR'])['AMT_CREDIT_SUM'].sum().reset_index().rename( index = str, columns = { 'AMT_CREDIT_SUM': 'TOTAL_CUSTOMER_CREDIT'})

    B = B.merge(grp1, on = ['SK_ID_CURR'], how = 'left')
    B = B.merge(grp2, on = ['SK_ID_CURR'], how = 'left')
    del grp1, grp2

    
    B['DEBT_CREDIT_RATIO'] = B['TOTAL_CUSTOMER_DEBT']/B['TOTAL_CUSTOMER_CREDIT']
    B.loc[B['TOTAL_CUSTOMER_CREDIT']==0,'DEBT_CREDIT_RATIO']=0 #ajouté

    del B['TOTAL_CUSTOMER_DEBT'], B['TOTAL_CUSTOMER_CREDIT']
    print('feature 8 ok')

    #FEATURE 9 - OVERDUE OVER DEBT RATIO
    #What fraction of total Debt is overdue per customer?
    #A high value could indicate a potential DEFAULT


    B['AMT_CREDIT_SUM_DEBT'] = B['AMT_CREDIT_SUM_DEBT'].fillna(0)
    B['AMT_CREDIT_SUM_OVERDUE'] = B['AMT_CREDIT_SUM_OVERDUE'].fillna(0)

    grp1 = B[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT']].groupby(by = ['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename( index = str, columns = { 'AMT_CREDIT_SUM_DEBT': 'TOTAL_CUSTOMER_DEBT'})
    grp2 = B[['SK_ID_CURR', 'AMT_CREDIT_SUM_OVERDUE']].groupby(by = ['SK_ID_CURR'])['AMT_CREDIT_SUM_OVERDUE'].sum().reset_index().rename( index = str, columns = { 'AMT_CREDIT_SUM_OVERDUE': 'TOTAL_CUSTOMER_OVERDUE'})

    B = B.merge(grp1, on = ['SK_ID_CURR'], how = 'left')
    B = B.merge(grp2, on = ['SK_ID_CURR'], how = 'left')
    del grp1, grp2

    B['OVERDUE_DEBT_RATIO'] = B['TOTAL_CUSTOMER_OVERDUE']/B['TOTAL_CUSTOMER_DEBT']
    B.loc[B['TOTAL_CUSTOMER_DEBT']==0,'OVERDUE_DEBT_RATIO']=0 #ajouté

    del B['TOTAL_CUSTOMER_OVERDUE'], B['TOTAL_CUSTOMER_DEBT']
    print('feature 9 ok')

    #FEATURE 10 - AVERAGE NUMBER OF LOANS PROLONGED
    B['CNT_CREDIT_PROLONG'] = B['CNT_CREDIT_PROLONG'].fillna(0)
    grp = B[['SK_ID_CURR', 'CNT_CREDIT_PROLONG']].groupby(by = ['SK_ID_CURR'])['CNT_CREDIT_PROLONG'].mean().reset_index().rename( index = str, columns = { 'CNT_CREDIT_PROLONG': 'AVG_CREDITDAYS_PROLONGED'})
    B = B.merge(grp, on = ['SK_ID_CURR'], how = 'left')
    print(B.shape)
    print('feature 10 ok')
    return B

def feat_eng(application_train, application_test): # de notebookd30915a6f4 (sur Kaggle)
#2eme partie du feature engeneering
    df=application_train.copy()
    df=df.append(application_test)


    df['DAYS_BIRTH'] = abs(df['DAYS_BIRTH'])
    df['family_members_more7']= np.where(df['CNT_FAM_MEMBERS']>7,'Oui','Non')
    #df['islowskilled_labour']= np.where(df['OCCUPATION_TYPE']=='Low-skill Laborers','Oui','Non')
    #df['is_Maternity_leave']= np.where(df['NAME_INCOME_TYPE'] =='Maternity leave' ,'Oui','Non')
    #df['is_unemployed']= np.where(df['NAME_INCOME_TYPE'] =='Unemployed' ,'Oui','Non')

    df['cnt_childern_more6']= np.where(df['CNT_CHILDREN'] > 6,'Oui','Non')
    
    df = df[df['CODE_GENDER']!='XNA']

    df['AGE_RANGE'] = df['DAYS_BIRTH'].apply(lambda x: get_age_label(x))
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['INCOME_TO_EMPLOYED_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_EMPLOYED']
    df.loc[df['DAYS_EMPLOYED']==0,'INCOME_TO_EMPLOYED_RATIO'] = 0 # ajouté
    df['INCOME_TO_BIRTH_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_BIRTH']
    df['ID_TO_BIRTH_RATIO'] = df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH']
    df['CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['CAR_TO_EMPLOYED_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    
    for function_name in ['min', 'max', 'mean', 'nanmedian', 'var']:
        feature_name = 'EXT_SOURCES_{}'.format(function_name.upper())
        df[feature_name] = eval('np.{}'.format(function_name))(
            df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)

    df['EXT_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['CREDIT_LENGTH'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']

    external_sources = df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH']]
    #imputer = SimpleImputer(strategy = 'median')
    #external_sources = imputer.fit_transform(external_sources)
    external_sources= pd.DataFrame(external_sources, columns = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                            'EXT_SOURCE_3', 'DAYS_BIRTH'])

    external_sources['SK_ID_CURR'] = df['SK_ID_CURR'].tolist()
    df = df.merge(external_sources, on = 'SK_ID_CURR', how = 'left')
    return df

def get_age_label(days_birth):
    """ Return the age group label (int). """
    age_years = -days_birth / 365
    if age_years < 27: return 1
    elif age_years < 40: return 2
    elif age_years < 50: return 3
    elif age_years < 65: return 4
    elif age_years < 99: return 5
    else: return 0

def feature_engineering(path, feat_eng1,feat_eng2,feat_eng3):
    ##1ère partie
    if not feat_eng1:
        bureau=pd.read_csv(path+'bureau.csv')
        B=create_bureau_features(bureau) #creation de 10 features
        B.to_csv(path+'bureau_avec_features.csv')
        bureau_avec_features=pd.read_csv(path+'bureau_avec_features.csv', index_col=0)
    else :
        bureau_avec_features=pd.read_csv(path+'bureau_avec_features.csv', index_col=0)
    print('ok 1ère partie')
    ##2eme partie
    if not feat_eng2:
        application_test=pd.read_csv(path+'application_test.csv')
        application_train=pd.read_csv(path+'application_train.csv')
        application=feat_eng(application_train, application_test)
        application.to_csv(path+'application_all.csv')
        application=pd.read_csv(path+'application_all.csv', index_col=0)
    else :
        application=pd.read_csv(path+'application_all.csv', index_col=0)
    print('ok 2eme partie')
    ##3eme partie
    if not feat_eng3:
        B2=bureau_avec_features.iloc[:,-8:-6].join(bureau_avec_features.iloc[:,-5:]).join(bureau_avec_features.iloc[:,0])
        B2=B2.drop_duplicates()
        del bureau_avec_features
        print('ok1')
        application_final=pd.merge(application, B2, on='SK_ID_CURR')
        application_final.to_csv(path+'application_final.csv')
        application_final=pd.read_csv(path+'application_final.csv', index_col=0)
        print('ok2')

def preparation_df(df):
    test_set=df[df['TARGET'].isnull()]
    train_set=df[df['TARGET'].notnull()]
    label=train_set['TARGET'] #label
    features=df.columns[\
        np.isin(df.columns,'TARGET', invert=True)] #features
    col_cat=df.loc[:,features]\
        .select_dtypes(exclude='number').columns #colonnes categorielles
    col_num=df.loc[:,features]\
        .select_dtypes('number').columns #colonnes numériques
    
    return train_set,test_set,label,col_cat,col_num,features

def preprocessing(col_num,col_cat, cat_col_cat):
    ohe=OneHotEncoder(sparse=False, handle_unknown='ignore', categories=cat_col_cat)
    col_cat_tr=pd.DataFrame(ohe.fit_transform(col_cat))

    ss=StandardScaler()
    col_num_tr=pd.DataFrame(ss.fit_transform(col_num))

    all_feat_ok=pd.concat([col_cat_tr,col_num_tr], axis=1)
    return all_feat_ok, ohe, ss

def imputation_valeurs_manquantes(df,col_num, col_cat):
    sp1=SimpleImputer(strategy='mean')
    col_num2=pd.DataFrame(sp1.fit_transform(df.loc[:,col_num]))
    col_num2.columns=col_num

    sp2=SimpleImputer(strategy='constant', fill_value='inconnu')
    col_cat2=pd.DataFrame(sp2.fit_transform(df.loc[:,col_cat]))
    col_cat2.columns=col_cat
    return col_num2, col_cat2

def scores(y_test,resu,proba,mod,df_final_test, graphs):
    f1=f1_score(y_test,resu)
    acc =accuracy_score(y_test,resu)
    mat=confusion_matrix(y_test,resu)
    a_u_c=roc_auc_score(y_test,proba)
    pre,rec,thr = precision_recall_curve(y_test,proba) 
    auc_pr=auc(rec,pre)
    print('Matrice de confusion :{}'.format(mat)) 
    print('Accuracy : {} ROC AUC : {} AUC Precision-Recall : {} F1 : {}\n'\
        .format(round(acc,3),round(a_u_c,3),round(auc_pr,3),round(f1,3)))
    if graphs==True:
        plot_roc_curve(mod,df_final_test,y_test)
        plt.show(block=False)
        plot_precision_recall_curve(mod,df_final_test,y_test)
        plt.show(block=False)
        plot_confusion_matrix(mod,df_final_test,y_test)
        plt.show(block=False)
    return acc, mat, a_u_c, f1, auc_pr

def nom_colonnes(col_cat_train,col_num_train):
    tab_nom_col_cat=[]
    tab_nom_col=[]
    for i in col_cat_train.columns:
        for j in col_cat_train[i].unique():
            tab_nom_col_cat.append('{}_{}'.format(i,j))
            tab_nom_col.append('{}_{}'.format(i,j))
    for i in col_num_train.columns:
        tab_nom_col.append(i)
    return tab_nom_col_cat,tab_nom_col

def modelisation2(df_final_train,y_train,df_final_test,y_test,meth,param_lr,graphs):
    ftwo_scorer = make_scorer(fbeta_score, beta=2)

    result =[]
    for i in meth: # Undersampling / Oversampling du dataset
        if i=='SMOTE': # 1 SMOTE
            smot=SMOTE(random_state=0)
            df_final_train_v2, y_train_v2 = smot.fit_resample\
                (np.array(df_final_train), np.array(y_train))
        elif i=='RandomUnderSampler':  # 2 RandomUnderSampler
            rus = RandomUnderSampler(random_state=0)
            df_final_train_v2, y_train_v2 = rus.fit_resample\
                (np.array(df_final_train), np.array(y_train))
        else : #Aucune \ Class_weight
            df_final_train_v2=df_final_train
            y_train_v2=y_train
    
        if i =='Class_weight':
            cw='balanced'
        else:
            cw=None

        est=LogisticRegression(max_iter=2000, class_weight=cw,random_state=0, solver='liblinear')
        param=param_lr
        gs=GridSearchCV(est,param,cv=3, scoring=ftwo_scorer,verbose=5)
        gs.fit(df_final_train_v2, y_train_v2)
        resu=gs.predict(df_final_test)
        proba=(gs.predict_proba(df_final_test)).T[1]
        best_params=gs.best_params_
        acc, mat, a_u_c, f1,roc_pr=scores(y_test,resu,proba,est,df_final_test, graphs) 
        # Scores
            
        resultat={'methode':i, 'Best_params':best_params,'Accuracy':acc,
        'ROC_AUC':a_u_c,'Precision_Recall_AUC' : roc_pr,'F1_score':f1,
        'Confusion_matrix':mat,'True Negative':mat[0,0],'True Positive':mat[1,1],
        'False Positive': mat[0,1],'False Negative': mat[1,0]}
        result.append(resultat)
        # Graphs

    result=pd.DataFrame(result)
    result['Recall']=result['True Positive']/(result['True Positive']+result['False Negative'])
    result['Precision']=result['True Positive']/(result['True Positive']+result['False Positive'])

    return result

def score_par_seuil(y_test,y_pred,seuil):
    tab_res=pd.DataFrame()
    for i in range(len(seuil)):
        valcol=[1 if n[1]>(seuil[i]/100) else 0 for n in y_pred]
        nomcol='pred_{}'.format(seuil[i])
        resul=pd.Series(valcol)
        resul.name=nomcol
        tab_res=tab_res.append(resul)
    tab_res=tab_res.T
    tab_res['TARGET']=y_test.values

    df_resu=[]

    for i in range(len(seuil)):
        confu=confusion_matrix(tab_res.iloc[:,-1],tab_res.iloc[:,i])
        tp=confu[1][1]
        tn=confu[0][0]
        fp=confu[0][1]
        fn=confu[1][0]
        resu={'Confusion_matrix':confu,'TP':tp,'TN':tn,'FP':fp,'FN':fn}
        df_resu.append(resu)
    df_resu=pd.DataFrame(df_resu)
    df_resu.index=seuil
    df_resu['Recall']=df_resu['TP']/(df_resu['TP']+df_resu['FN'])
    df_resu['Precision']=df_resu['TP']/(df_resu['TP']+df_resu['FP'])
    df_resu['Accuracy']=(df_resu['TP']+df_resu['TN'])/(df_resu['TP']+df_resu['FP']+df_resu['FN']+df_resu['TN'])
    df_resu['F1']=(2 * df_resu['Precision'] * df_resu['Recall']) / (df_resu['Precision'] + df_resu['Recall'])
    df_resu.sort_values('F1', ascending=False)
    a_u_c=roc_auc_score(y_test,y_pred.T[1])
    pre,rec,_ = precision_recall_curve(y_test,y_pred.T[1]) 
    auc_pr=auc(rec,pre)

    
    return tab_res,df_resu,a_u_c,auc_pr


def post_feat_eng(application_final):

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

    for i in col_boo:
        application_final[i]=application_final[i].map({0:'Oui',1:'Non'})
    
    application_final.loc[application_final['DAYS_EMPLOYED']>0,'DAYS_EMPLOYED']=np.nan
    dict_name_income_type={
    'Working':'Working', 
    'State servant':'State servant', 
    'Commercial associate':'other',
    'Pensioner':'Pensioner',
    'Unemployed':'Unemployed', 
    'Student':'other', 
    'Businessman':'other', 
    'Maternity leave':'Maternity leave'}

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
    'Low-skill Laborers':'Low-skill Laborers', 
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

    variables_supprimees=['AGE_RANGE','APARTMENTS_MEDI','YEARS_BUILD_MODE']#,'SK_ID_CURR']

    variables_supprimees_2=['HOUR_APPR_PROCESS_START',
                'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'FLAG_EMAIL', 'FLAG_PHONE',
                 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
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
                'FONDKAPREMONT_MODE', 'EMERGENCYSTATE_MODE' ,'OWN_CAR_AGE','CAR_TO_EMPLOYED_RATIO', 
                'CAR_TO_BIRTH_RATIO', 'EXT_SOURCES_PROD', 'APARTMENTS_AVG', 'APARTMENTS_MODE',
                'APARTMENTS_MEDI','ENTRANCES_AVG', 'ENTRANCES_MEDI', 'LIVINGAREA_AVG', 'FLOORSMAX_MEDI', 
                'FLOORSMAX_AVG','FLOORSMAX_MODE', 'YEARS_BEGINEXPLUATATION_MEDI', 'TOTALAREA_MODE',
                'EXT_SOURCE_1_y','EXT_SOURCE_2_y','EXT_SOURCE_3_y','DAYS_BIRTH_y',
                'FLAG_DOCUMENT_2','AMT_REQ_CREDIT_BUREAU_HOUR','FLAG_DOCUMENT_12','FLAG_DOCUMENT_4',
                'FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_5','WALLSMATERIAL_MODE',
                'FLAG_DOCUMENT_19','EXT_SOURCES_MEAN','FLAG_DOCUMENT_10','is_unemployed','is_Maternity_leave']


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
                'NAME_TYPE_SUITE': 'ACCOMPAGNATEUR',
                'NAME_INCOME_TYPE':'TYPE_DE_REVENUS', 
                'NAME_EDUCATION_TYPE':'NIVEAU_D_ETUDES',
                'NAME_FAMILY_STATUS':'STATUT FAMILIAL',
                'NAME_HOUSING_TYPE':'LOGEMENT_ACTUEL',
                'REGION_POPULATION_RELATIVE':'POPULATION_DE_LA_REGION_NORMALISEE',
                'DAYS_BIRTH_x':'AGE', 
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
                'TARGET':'TARGET',
                'CNT_CHILDREN':'NOMBRE_D_ENFANTS', 
                'CNT_FAM_MEMBERS':'MEMBRES_DE_LA_FAMILLE',
                'FLAG_OWN_REALTY':'POSSEDE_UN_LOGEMENT',
                'FLAG_OWN_CAR':'POSSEDE_UNE_VOITURE'}

    application_final=application_final.rename(columns=nom_colonnes)
    application_final['AGE']=application_final['AGE']//365.25

    return application_final
