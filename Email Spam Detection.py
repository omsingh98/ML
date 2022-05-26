#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv("spam.csv")
df.head()


# In[2]:


df.groupby('Category').describe()


# In[3]:


df["spam"] = df["Category"].apply(lambda x: 1 if x =="spam" else 0)
df.head()


# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size = 0.25)
len(X_train)


# In[6]:


from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray()[:3]


# In[7]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count, y_train)


# In[8]:


emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]
emails_count = v.transform(emails)
model.predict(emails_count)


# In[9]:


X_test_count = v.transform(X_test)
model.score(X_test_count, y_test)


# In[10]:


from sklearn.pipeline import Pipeline
clf = Pipeline([
    ("Vectorizer", CountVectorizer()),
    ("nb", MultinomialNB())
])


# In[11]:


clf.fit(X_train, y_train)


# In[12]:


clf.score(X_test, y_test)


# In[13]:


clf.predict(emails)


# In[ ]:




