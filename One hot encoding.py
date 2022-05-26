#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
df = pd.read_csv("homepricesone.csv")


# In[27]:


df


# In[28]:


dummies = pd.get_dummies(df.town)
dummies


# In[29]:


merged = pd.concat([df,dummies], axis = "columns")
merged


# In[30]:


final = merged.drop(["town","west windsor"], axis = "columns")
final


# In[31]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[33]:


X = final.drop("price", axis= "columns")


# In[34]:


X


# In[35]:


y = final.price
y


# In[36]:


model.fit(X,y)


# In[37]:


model.predict([[2800,0,1]])


# In[38]:


model.score(X,y)


# In[39]:


df


# In[14]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[40]:


dfle = df
dfle.town = le.fit_transform(dfle.town)
dfle


# In[41]:


X = dfle[['town','area']].values


# In[42]:


X


# In[43]:


X = dfle[["town","area"]].values
X


# In[44]:


y = dfle.price.values
y


# In[45]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('town', OneHotEncoder(), [0])], remainder = 'passthrough')


# In[46]:


X = ct.fit_transform(X)
X


# In[47]:


X = X[:,1:]


# In[48]:


X


# In[49]:


model.fit(X,y)


# 

# In[50]:


model.predict([[0,1,3400]])


# In[ ]:




