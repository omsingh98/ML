#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
from sklearn.datasets import load_digits


# In[52]:


dataset = load_digits()
dataset.keys()


# In[53]:


datasets.data[0]


# In[54]:


datasets.data[0].reshape(8,8)


# In[55]:


from matplotlib import pyplot as plt


# In[56]:


plt.gray()
plt.matshow(dataset.data[50].reshape(8,8))


# In[57]:


datasets.target


# In[58]:


df = pd.DataFrame(dataset.data)
df.describe()


# In[59]:


X = df
y= dataset.target


# In[60]:


y


# In[61]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)
x_scaled


# In[62]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_scaled,y,  test_size = 0.2)


# In[63]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)


# In[66]:


from sklearn.decomposition import PCA
pca = PCA(0.95)

x_pca = pca.fit_transform(X)
x_pca.shape


# In[65]:


X.shape


# In[67]:


X_train_pca, X_test_pca, y_train, y_test = train_test_split(x_pca,y,  test_size = 0.2)


# In[68]:


model = LogisticRegression()
model.fit(X_train_pca, y_train)
model.score(X_test_pca, y_test)


# In[ ]:




