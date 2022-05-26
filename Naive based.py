#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
df = pd.read_csv("titanic.csv")
df.head()


# In[37]:


df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
df.head()


# In[38]:


inputs = df.drop('Survived',axis='columns')
target = df.Survived


# In[39]:


dummies = pd.get_dummies(inputs.Sex)
dummies.head(3)


# In[40]:


inputs = pd.concat([inputs,dummies],axis='columns')
inputs.head(3)


# In[42]:


inputs.drop(['Sex','male'],axis='columns',inplace=True)
inputs.head(3)


# In[43]:


inputs.columns[inputs.isna().any()]


# In[45]:


inputs.Age[:10]


# In[47]:


inputs.Age = inputs.Age.fillna(inputs.Age.mean())
inputs.Age[:10]


# In[48]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size = 0.2)


# In[50]:


len(X_train)


# In[51]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()


# In[53]:


model.fit(X_train, y_train)


# In[54]:


model.score(X_test, y_test)


# In[55]:


model.predict(X_test[:10])


# In[ ]:




