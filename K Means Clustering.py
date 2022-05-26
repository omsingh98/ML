#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt


# In[3]:


df = pd.read_csv("income.csv")
df.head()


# In[3]:


plt.scatter(df["Age"],df["Income($)"])


# In[9]:


km =KMeans(n_clusters = 3)
km


# In[10]:


y_predicted = km.fit_predict(df[["Age","Income($)"]])
y_predicted


# In[11]:


df["cluster"] = y_predicted
df.head()


# In[12]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.scatter(df1.Age,df1["Income($)"],color = "green")
plt.scatter(df2.Age,df2["Income($)"],color = "red")
plt.scatter(df3.Age,df3["Income($)"],color = "black")

plt.xlabel("Age")
plt.ylabel("Income")


# In[13]:


scaler = MinMaxScaler()

scaler.fit(df[['Income($)']])
df['Income($)'] = scaler.transform(df[['Income($)']])

scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])
df


# In[14]:


km =KMeans(n_clusters = 3)
y_predicted = km.fit_predict(df[["Age","Income($)"]])
y_predicted


# In[15]:


df["cluster"] = y_predicted
df.head()


# In[16]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.scatter(df1.Age,df1["Income($)"],color = "green")
plt.scatter(df2.Age,df2["Income($)"],color = "red")
plt.scatter(df3.Age,df3["Income($)"],color = "black")

plt.xlabel("Age")
plt.ylabel("Income")


# In[ ]:




