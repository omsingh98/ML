#!/usr/bin/env python
# coding: utf-8

# In[37]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from sklearn.datasets import load_digits
digits = load_digits()


# In[44]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3)


# In[45]:


lr = LogisticRegression()
lr.fit(X_train,y_train)
lr.score(X_test, y_test)


# In[46]:


svm = SVC()
svm.fit(X_train,y_train)
svm.score(X_test, y_test)


# In[47]:


rf = RandomForestClassifier()
rf.fit(X_train,y_train)
rf.score(X_test, y_test)


# In[58]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
kf


# In[59]:


for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(train_index, test_index)


# In[60]:


def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


# In[61]:


get_score(LogisticRegression(), X_train, X_test, y_train, y_test)


# In[62]:


from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits=3)


# In[63]:


scores_logistic = []
scores_svm = []
scores_rf = []

for train_index, test_index in folds.split(digits.data,digits.target):
    X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index], digits.target[train_index], digits.target[test_index]
    scores_logistic.append(get_score(LogisticRegression(solver='liblinear',multi_class='ovr'), X_train, X_test, y_train, y_test))  
    scores_svm.append(get_score(SVC(gamma='auto'), X_train, X_test, y_train, y_test))
    scores_rf.append(get_score(RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test))


# In[64]:


scores_logistic


# In[65]:


scores_svm


# In[66]:


scores_rf


# In[67]:


from sklearn.model_selection import cross_val_score
cross_val_score(LogisticRegression(), digits.data, digits.target)


# In[68]:


cross_val_score(SVC(), digits.data, digits.target)


# cross_val_score(RandomForestClassifier(n_estimators=40), digits.data, digits.target)

# In[ ]:





# In[ ]:




