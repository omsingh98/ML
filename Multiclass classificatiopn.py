#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


# In[3]:


digits = load_digits()


# In[4]:


dir(digits)


# In[21]:


digits.data[0]


# In[22]:


plt.gray()
plt.matshow(digits.images[0])


# In[7]:


digits.target[0]


# In[8]:


from sklearn.model_selection import train_test_split


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.2)


# In[24]:


len(X_train)


# In[25]:


len(X_test)


# In[26]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[27]:


model.fit(X_train, y_train)


# In[30]:


model.score(X_test, y_test)


# In[28]:


model.predict([digits.data[67]])


# In[29]:


digits.target[67]


# In[ ]:




