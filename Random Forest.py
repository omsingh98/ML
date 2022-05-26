#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.datasets import load_digits
digits = load_digits()


# In[2]:


dir(digits)


# In[4]:


import matplotlib.pyplot as plt
plt.gray()
for i in range(4):
    plt.matshow(digits.images[i])


# In[6]:


df = pd.DataFrame(digits.data)
df.head()


# In[8]:


df["targets"] = digits.target
df.head()


# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(['targets'], axis = "columns"), digits.target,test_size = 0.2)


# In[10]:


len(X_test)


# In[11]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)


# In[12]:


model.score(X_test, y_test)


# In[20]:


X_test


# In[ ]:





# In[ ]:




