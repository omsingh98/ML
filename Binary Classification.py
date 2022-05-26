#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("insurance_data.csv")
df.head()


# In[3]:


plt.scatter(df.age,df.bought_insurance)


# In[4]:


df.shape


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


x_train, x_test, y_train, y_test = train_test_split(df[["age"]],df.bought_insurance,test_size=0.1)


# In[9]:


x_test


# In[10]:


x_train


# In[11]:


from sklearn.linear_model import LogisticRegression


# In[12]:


model = LogisticRegression()


# In[13]:


model.fit(x_train,y_train)


# In[14]:


model.predict(x_test)


# In[15]:


model.score(x_test,y_test)


# In[ ]:





# In[ ]:




