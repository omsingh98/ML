#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()


# In[3]:


iris.feature_names


# In[4]:


iris.target_names


# In[5]:


df = pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()


# In[6]:


df['target'] = iris.target
df.head()


# In[7]:


df[df.target==1].head()


# In[8]:


df[df.target==2].head()


# In[9]:


df['flower_name'] =df.target.apply(lambda x: iris.target_names[x])
df.head()


# In[10]:


df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]


# In[11]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color="blue",marker='.')


# In[13]:


from sklearn.model_selection import train_test_split
X = df.drop(['target','flower_name'], axis='columns')
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
len(X_train)
len(X_test)


# In[14]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)


# In[15]:


knn.fit(X_train, y_train)


# In[16]:


knn.score(X_test, y_test)


# In[17]:


knn.predict([[4.8,3.0,1.5,0.3]])


# In[ ]:




