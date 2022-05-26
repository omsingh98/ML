#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()


# In[10]:


dir(iris)


# In[11]:


iris.feature_names


# In[12]:


df = pd.DataFrame(iris.data, columns =iris.feature_names)
df.head()


# In[13]:


df['target'] = iris.target
df.head()


# In[14]:


iris.target_names


# In[17]:


df[df.target==0].head()


# In[19]:


df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
df.tail()


# In[24]:


from matplotlib import pyplot as plt


# In[20]:


df0 = df[df.target==0]
df1 = df[df.target==1]
df2 = df[df.target==2]


# In[22]:


df1.head()


# In[25]:


plt.xlabel('sapel length(cm)')
plt.ylabel('sapel widht (cm)')

plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color = 'green', marker = '+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color = 'red', marker = '+')


# In[26]:


plt.xlabel('petal length(cm)')
plt.ylabel('petal widht (cm)')

plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color = 'green', marker = '+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color = 'red', marker = '+')


# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


X = df.drop(["target", "flower_name"], axis = "columns")
X


# In[29]:


y = df.target
y


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# In[31]:


len(X_train)


# In[32]:


len(X_test)


# In[33]:


from sklearn.svm import SVC
model = SVC()


# In[34]:


model.fit(X_train, y_train)


# In[36]:


model.predict([[5.1,3.5,1.4,0.2]])


# In[35]:


model.score(X_train, y_train)


# In[ ]:





# In[ ]:




