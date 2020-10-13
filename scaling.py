#!/usr/bin/env python
# coding: utf-8

# Exercises
# Do your work for these exercises in a jupyter notebook named scaling. Use the telco dataset. Once you are finished, you may wish to repeat the exercises on another dataset for additional practice.
# 
# Apply the scalers we talked about in this lesson to your data and visualize the results in a way that can .
# Apply the .inverse_transform method to your scaled data. Is the resulting dataset the exact same as the original data?
# Read the documentation for sklearn's QuantileTransformer. Use normal for the output_distribution and apply this scaler to your data. Visualize the result of your data scaling.
# Use the QuantileTransformer, but omit the output_distribution argument. Visualize your results. What do you notice?
# Based on the work you've done, choose a scaling method for your dataset. Write a function within your prepare.py that accepts as input the train, validate, and test data splits, and returns the scaled versions of each. Be sure to only learn the parameters for scaling from your training data!

# In[1]:


import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pydataset
from sklearn.model_selection import train_test_split
import prepare
np.random.seed(123)


# In[79]:


df= prepare.clean_telco()


# In[80]:


df.info()


# In[81]:


df


# In[82]:


df['total_charges'] = pd.to_numeric(df['total_charges'],errors='coerce')
df.info()


# In[22]:


df.plot.scatter(y='monthly_charges', x='years_tenure')


# In[70]:


df.plot.scatter(y='total_charges', x='years_tenure')


# In[10]:


print(df.total_charges.describe())


# In[27]:


train_and_validate, test = train_test_split(df, test_size=.12, random_state=123)
train, validate = train_test_split(train_and_validate, test_size=.12, random_state=123)


# In[28]:


train.shape, test.shape, validate.shape


# Visualizing Scalers

# In[74]:


visualize_scaler(sklearn.preprocessing.MinMaxScaler(), 'Min-Max Scaling')


# In[29]:


# 1. create the object
scaler = sklearn.preprocessing.MinMaxScaler()


# In[30]:


# 2. fit the object
scaler.fit(train[['monthly_charges']])


# In[33]:


# 3. use the object
train['monthly_charges_scaled'] = scaler.transform(train[['monthly_charges']])
test['monthly_charges_scaled'] = scaler.transform(test[['monthly_charges']])
validate['monthly_charges_scaled'] = scaler.transform(validate[['monthly_charges']])


# In[34]:


plt.figure(figsize=(13, 6))
plt.subplot(121)
train.monthly_charges.plot.hist(title='Original')
plt.subplot(122)
train.monthly_charges_scaled.plot.hist(title='Min-Max Scaled')


# In[36]:


scaler = sklearn.preprocessing.StandardScaler()
# 2. fit the object
scaler.fit(train[['monthly_charges']])
# 3. use the object
train['monthly_charges_scaled'] = scaler.transform(train[['monthly_charges']])
test['monthly_charges_scaled'] = scaler.transform(test[['monthly_charges']])
validate['monthly_charges_scaled'] = scaler.transform(validate[['monthly_charges']])


# In[37]:


plt.figure(figsize=(13, 6))
plt.subplot(121)
train.monthly_charges.plot.hist(title='Original')
plt.subplot(122)
train.monthly_charges_scaled.plot.hist(title='Min-Max Scaled')


# non-linear scaler

# In[39]:


# 1. create the object
scaler = sklearn.preprocessing.QuantileTransformer(output_distribution='normal')
# 2. fit the object
scaler.fit(train[['monthly_charges']])
# 3. use the object
train['monthly_charges_scaled'] = scaler.transform(train[['monthly_charges']])
test['monthly_charges_scaled'] = scaler.transform(test[['monthly_charges']])
validate['monthly_charges_scaled'] = scaler.transform(validate[['monthly_charges']])

plt.figure(figsize=(13, 6))
plt.subplot(121)
train.monthly_charges.plot.hist(title='Original')
plt.subplot(122)
train.monthly_charges_scaled.plot.hist(title='Quantile Transformed to Normal')


# In[42]:


train = train.drop(columns='monthly_charges_scaled')


# In[43]:


def add_scaled_columns(train, validate, test, scaler, columns_to_scale):
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    scaler.fit(train[columns_to_scale])

    train = pd.concat([
        train,
        pd.DataFrame(scaler.transform(train[columns_to_scale]), columns=new_column_names, index=train.index),
    ], axis=1)
    validate = pd.concat([
        validate,
        pd.DataFrame(scaler.transform(validate[columns_to_scale]), columns=new_column_names, index=validate.index),
    ], axis=1)
    test = pd.concat([
        test,
        pd.DataFrame(scaler.transform(test[columns_to_scale]), columns=new_column_names, index=test.index),
    ], axis=1)
    
    return train, validate, test


# In[49]:


train, validate, test = add_scaled_columns(
    train,
    validate,
    test,
    scaler=sklearn.preprocessing.MinMaxScaler(),
    columns_to_scale=['monthly_charges']
    ,
)


# In[75]:


train[['monthly_charges']]


# In[83]:


scaler = sklearn.preprocessing.MinMaxScaler()

scaler.fit(train[['monthly_charges', 'total_charges']])

train_scaled = scaler.transform(train[['monthly_charges', 'total_charges']])
train_scaled = pd.DataFrame(train_scaled, columns=['monthly_charges_scaled', 'total_charges_scaled'])
train_scaled

scaler.inverse_transform(train_scaled[['monthly_charges_scaled', 'total_charges_scaled']])


# In[1]:





# In[2]:





# In[3]:





# In[ ]:




