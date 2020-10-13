#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import acquire


# <h1 style='color: darkorchid'><font face='chalkduster'> Wrange</h1>
# <p style= 'color: darkmagenta'><font face='chalkduster'> We will start with mall_customers database.
# <ul style= 'color: darkmagenta'><font face='chalkduster'>
# <li>acquire: verify our acquire module is working</li>
# <li>summarize our data</li>
# <li>plot histograms + boxplots</li>
# <li>na's</li>
# <li>outliers</li>
# <li>astype()</li>
# <li>pd.cut()</li>
# </ul></p>
# 

# <h2 style='color:orchid'><font face='chalkduster'> Acquire</h2>

# In[2]:


df=acquire.get_mall_data()


# <h2 style='color:orchid'><font face='chalkduster'> Summarize </h2>

# In[3]:


df.info()


# In[4]:


df.head()


# In[5]:


df.dtypes


# <p style= 'color: darkmagenta'><font face='chalkduster'>Takeaways</p>
# <ul style= 'color: darkmagenta'><font face='chalkduster'>
# <li>encode gender</li>
# <li>drop customer_id</li>
# <li>define our target variable: spending_score</li>

# <h2 style='color:orchid'><font face='chalkduster'>Plot distributions </h2>

# In[6]:


df.columns
for col in ['age', 'annual_income', 'spending_score']:
    plt.hist(df[col], color='violet')
    plt.show()


# Boxplots

# In[7]:


plt.figure(figsize=(12, 10))
sns.boxplot(data= df[['age', 'annual_income', 'spending_score']])
plt.title('Columns Box Plot')


# In[8]:


df.isna().sum()


# Takeaway: no nulls in our data set

# In[9]:


df['is_female']= (df.gender== 'Female').astype('int')
df.head()


# In[10]:


from sklearn.model_selection import train_test_split

train_and_validate, test = train_test_split(df, test_size=.15, random_state=123)
train, validate= train_test_split(train_and_validate, test_size=.15, random_state= 123)
print('train', train.shape)
print('test', test.shape)
print('validate', validate.shape)


# In[11]:


df=acquire.get_mall_data()

def prep_mall_data(df):
    '''Takes the acquired mall data, does data prep, and returns 
    train, test, validate data splits'''
    df['is_female']= (df.gender== 'Female').astype('int')
    train_and_validate, test = train_test_split(df, test_size=.15, random_state=123)
    train, validate= train_test_split(train_and_validate, test_size=.15, random_state= 123)
    return train, test, validate


# <h2 style='color:orchid'><font face='chalkduster'> Acquire</h2>

# In[3]:


df= acquire.new_telco_data()


# In[5]:


df.head()


# In[8]:


df= df.drop(columns=['gender', 'senior_citizen', 'partner', 'tech_support','device_protection','online_backup','dependents', 'phone_service', 'multiple_lines', 'internet_service_type_id', 'online_security','streaming_tv', 'streaming_movies', 'paperless_billing', 'payment_type_id', 'churn', 'contract_type_id'])


# In[11]:


print(df.isnull().sum())


# Takeaway: There is no nulls.

# In[12]:


df.describe()


# In[ ]:


def wrangle_grades():
    grades = pd.read_csv("student_grades.csv")
    grades.drop(columns="student_id", inplace=True)
    grades.replace(r"^\s*$", np.nan, regex=True, inplace=True)
    df = grades.dropna().astype("int")
    return df


# In[ ]:


def wrangle_telco():
    """
    Obtaines a sql queries the telco_churn database
    Returns a clean df with four columns:
    customer_id(object), monthly_charges(float), tenure(int), total_charges(float)
    """
    df = get_data_from_sql()
    df.tenure.replace(0, 1, inplace=True)
    df.total_charges = df.total_charges.replace(" ", np.nan)
    df.total_charges = df.total_charges.fillna(df.monthly_charges)
    df.total_charges = df.total_charges.astype(float)

    train_and_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_and_validate, test_size=.2, random_state=123)

    return scale_telco_data(train, test, validate)


# In[1]:





# In[2]:





# In[3]:





# In[ ]:




