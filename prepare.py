#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

from acquire import get_titanic_data, get_iris_data


# In[3]:


iris = get_iris_data()
iris.head()


# In[4]:


iris = iris.drop(columns='species_id')
iris.head(2)


# In[5]:


iris = iris.rename(columns={'species_name': 'species'})
iris.head(2)


# In[6]:


species_dummies = pd.get_dummies(iris.species, drop_first=True)
species_dummies.head(3)


# In[7]:


iris = pd.concat([iris, species_dummies], axis=1)
iris.head()


# In[11]:


def prep_iris(cached=True):
    '''
    This function acquires and prepares the iris data from a local csv, default.
    Passing cached=False acquires fresh data from Codeup db and writes to csv.
    Returns the iris df with dummy variables encoding species.
    '''
    
    # use my aquire function to read data into a df from a csv file
    df = get_iris_data()
    
    # drop and rename columns
    df = df.drop(columns='species_id').rename(columns={'species_name': 'species'})
    
    # create dummy columns for species
    species_dummies = pd.get_dummies(df.species, drop_first=True)
    
    # add dummy columns to df
    df = pd.concat([df, species_dummies], axis=1)
    
    return df


# In[12]:


iris = prep_iris()
iris.sample(7)


# In[13]:


## titanic data
titanic = get_titanic_data()
titanic.head()


# In[14]:


## handles NAN
titanic[titanic.embark_town.isnull()]


# In[15]:


titanic[titanic.embarked.isnull()]


# In[41]:


titanic['is_male']= titanic.sex == 'male'


# In[42]:


# using the complement operator, ~, to return the inverse of our instance above. Return everything but the null values.

titanic = titanic[~titanic.embarked.isnull()]
titanic.info()


# In[17]:


titanic = titanic.drop(columns='deck')
titanic.info()


# In[18]:


titanic_dummies = pd.get_dummies(titanic.embarked, drop_first=True)
titanic_dummies.sample(10)


# In[19]:


titanic = pd.concat([titanic, titanic_dummies], axis=1)
titanic.head()


# In[20]:


train_validate, test = train_test_split(titanic, test_size=.2, 
                                        random_state=123, 
                                        stratify=titanic.survived)


# In[21]:


train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.survived)


# In[23]:


print(f'train -> {train.shape}')
print(f'validate -> {validate.shape}')
print(f'test -> {test.shape}')


# In[24]:


def titanic_split(df):
    '''
    This function performs split on titanic data, stratify survived.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.survived)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.survived)
    return train, validate, test


# In[25]:


train, validate, test = titanic_split(titanic)


# In[26]:


print(f'train -> {train.shape}')
print(f'validate -> {validate.shape}')
print(f'test -> {test.shape}')


# In[27]:


train.head(2)


# In[28]:


# Create the imputer object.

imputer = SimpleImputer(strategy = 'mean')


# In[29]:


# Fit the imputer to train and transform.

train['age'] = imputer.fit_transform(train[['age']])


# In[30]:


# quick check

train['age'].isnull().sum()


# In[31]:


# Transform the validate and test df age columns

validate['age'] = imputer.transform(validate[['age']])
test['age'] = imputer.transform(test[['age']])


# In[32]:


def impute_mean_age(train, validate, test):
    '''
    This function imputes the mean of the age column into
    observations with missing values.
    Returns transformed train, validate, and test df.
    '''
    # create the imputer object with mean strategy
    imputer = SimpleImputer(strategy = 'mean')
    
    # fit on and transform age column in train
    train['age'] = imputer.fit_transform(train[['age']])
    
    # transform age column in validate
    validate['age'] = imputer.transform(validate[['age']])
    
    # transform age column in test
    test['age'] = imputer.transform(test[['age']])
    
    return train, validate, test


# In[35]:


def prep_titanic(cached=True):
    '''
    This function reads titanic data into a df from a csv file.
    Returns prepped train, validate, and test dfs
    '''
    # use my acquire function to read data into a df from a csv file
    df = get_titanic_data()
    
    # drop rows where embarked/embark town are null values
    df = df[~df.embarked.isnull()]
    
    # encode embarked using dummy columns
    titanic_dummies = pd.get_dummies(df.embarked, drop_first=True)
    # join dummy columns back to df
    df = pd.concat([df, titanic_dummies], axis=1)
    
    # drop the deck column
    df = df.drop(columns='deck')
    df['is_male']= titanic.sex == 'male'
    
    # split data into train, validate, test dfs
    train, validate, test = titanic_split(df)
    
    # impute mean of age into null values in age column
    train, validate, test = impute_mean_age(train, validate, test)
    
    return train, validate, test


# In[36]:


train, validate, test = prep_titanic()


# In[37]:


print(f'train -> {train.shape}')
print(f'validate -> {validate.shape}')
print(f'test -> {test.shape}')


# In[ ]:




