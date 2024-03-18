#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


car_df = pd.read_csv('Car_sales.csv')
pharm_df = pd.read_excel('data pharm 2.xlsx')


# In[3]:


car_df.tail()


# In[4]:


car_df.describe()


# In[5]:


car_df.Sales


# In[11]:


car_df.Sales.describe()


# In[12]:


sns.boxplot(y=car_df.Sales)

plt.title("Distribution of Car Sales")
plt.ylabel('Sales Statistics')

plt.show()


# In[13]:


sales_25 = np.percentile(car_df.Sales, 25)
sales_75 = np.percentile(car_df.Sales, 75)
sales_25, sales_75


# In[14]:


iqr = sales_75 - sales_25
iqr


# In[15]:


# 75 percentile + (1.5 * IQR)
upper_whisker = sales_75 + (1.5*iqr)

# 25 percentile + (1.5 * IQR)
lower_whisker = sales_25 - (1.5*iqr)

upper_whisker, lower_whisker


# In[20]:


# total number of records from dataframe that are 
# less than lower whisker value
sum(car_df.Sales < lower_whisker)


# In[22]:


# total number of records from dataframe that are 
# greater than upper whisker value
sum(car_df.Sales > upper_whisker)


# In[26]:


car_no_outlier_df = car_df[~(car_df.Sales > upper_whisker)]
car_no_outlier_df.head()


# In[28]:


sns.boxplot(car_no_outlier_df.Sales)


# In[29]:


car_no_outlier_df.describe()


# PREDICTIVE ANALYSIS - MACHINE LEARNING (ML)

# In[32]:


from sklearn import datasets


# In[36]:


cancer_data = datasets.load_breast_cancer()


# In[37]:


dir(cancer_data)


# In[42]:


print(cancer_data.DESCR)


# In[55]:


data = cancer_data.data
data[:2, :]


# In[56]:


features = cancer_data.feature_names
features


# In[59]:


cancer_df = pd.DataFrame(data, columns=features)
cancer_df.head()


# In[60]:


cancer_df.info()


# In[61]:


dir(cancer_data)


# In[62]:


cancer_data['target_names'], cancer_data['target']


# In[64]:


cancer_df['target'] = cancer_data['target']
cancer_df.head()


# In[66]:


target_names = []
for value in cancer_df.target: 
    if value == 0:
        target_names.append('Malignant')
    else:
        target_names.append('Benign')
        
cancer_df['target_names'] = target_names


# In[67]:


cancer_df.head()


# In[ ]:




