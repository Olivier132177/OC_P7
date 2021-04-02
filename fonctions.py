import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

def create_bureau_features(bureau): # de HOME CREDIT - BUREAU DATA - FEATURE ENGINEERING sur Kaggle
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


#    B['OVERDUE_DEBT_RATIO'] = B['TOTAL_CUSTOMER_OVERDUE']/B['TOTAL_CUSTOMER_DEBT']

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
    df=application_train.copy()
    df=df.append(application_test)


    df['DAYS_BIRTH'] = abs(df['DAYS_BIRTH'])
    df['family_members_more7']= np.where(df['CNT_FAM_MEMBERS']>7,1,0)
    df['islowskilled_labour']= np.where(df['OCCUPATION_TYPE']=='Low-skill Laborers',1,0)
    df['is_Maternity_leave']= np.where(df['NAME_INCOME_TYPE'] =='Maternity leave' ,1,0)
    df['is_unemployed']= np.where(df['NAME_INCOME_TYPE'] =='Unemployed' ,1,0)

    df['cnt_childern_more6']= np.where(df['CNT_CHILDREN'] > 6,1,0)
    plt.style.use('fivethirtyeight')

    # Plot the distribution of ages in years
    plt.hist(df['DAYS_BIRTH'] / 365, edgecolor = 'k', bins = 25)
    plt.title('Age of Client'); plt.xlabel('Age (years)'); plt.ylabel('Count')


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
    drop_list = [
            'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'HOUR_APPR_PROCESS_START',
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
            'FLOORSMAX_AVG','FLOORSMAX_MODE', 'YEARS_BEGINEXPLUATATION_MEDI', 'TOTALAREA_MODE'
        ]
    #df.drop(drop_list,axis=1,inplace=True)
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
