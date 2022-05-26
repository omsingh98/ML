#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
df = pd.read_csv("salaries.csv")
df.head()


# In[23]:


inputs = df.drop("salary_more_then_100k", axis = "columns")
targets = df["salary_more_then_100k"]


# In[36]:


inputs


# In[26]:


from sklearn.preprocessing import LabelEncoder


# In[27]:


le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()


# In[29]:


inputs["company_n"] = le_company.fit_transform(inputs["company"])
inputs["job_n"] = le_job.fit_transform(inputs["job"])
inputs["degree_n"] = le_degree.fit_transform(inputs["degree"])
inputs.head()


# In[30]:


inputs_n = inputs.drop(["job","degree", "company"], axis = "columns")
inputs_n


# In[31]:


from sklearn import tree


# In[32]:


model = tree.DecisionTreeClassifier()


# In[34]:


model.fit(inputs_n,targets)


# models.score(inputs_n, targets)

# In[35]:


model.score(inputs_n,targets)


# In[40]:


model.predict([[1,2,1]])


# In[ ]:




