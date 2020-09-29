import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
# ignore warnings
import warnings
warnings.filterwarnings("ignore")
from acquire import get_telco_data
import acquire
df=acquire.get_mall_data()

def prep_mall_data(df):
    '''Takes the acquired mall data, does data prep, and returns 
    train, test, validate data splits'''
    df['is_female']= (df.gender== 'Female').astype('int')
    train_and_validate, test = train_test_split(df, test_size=.15, random_state=123)
    train, validate= train_test_split(train_and_validate, test_size=.15, random_state= 123)
    return train, test, validate

def clean_telco(cached=False):
    df = get_telco_data()
    # use my aquire function to read data into a df from a csv file
    # drop duplicates
    df.drop_duplicates(inplace=True)
    # drop and rename columns
    df= df.drop(columns=['payment_type_id','contract_type_id.1', 'contract_type_id','internet_service_type_id'])
    df['years_tenure'] = df.tenure / 12
    df['has_streaming']= df["streaming_tv" or "streaming_movies"] == 'Yes'
    df['is_family']=df["partner" or "dependents"] == 'Yes'
    df['has_phones']= df['phone_service' or 'multiple_lines']== 'Yes'
    df['has_security_features']= df['online_security' or 'online_backup'] =='Yes'
    df['years_tenure'] = df.tenure / 12
    # create dummy columns for churn
    telco_dummies = pd.get_dummies(df.churn, drop_first=True)
    # add dummy columns to df
    df = pd.concat([df, telco_dummies], axis=1)
    # rename dummy columns
    df= df.rename(columns={'Yes': 'is_churn'})
    
    return df
#combining my split, train, test data and my clean data into one dataframe
def prep_telco_data():
    df = clean_telco()
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.is_churn)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.is_churn)
    return train, validate, test