#!/usr/bin/env python
# coding: utf-8

# In[45]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


# In[46]:


(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()


# In[47]:


len(X_train)


# In[48]:


X_train[0].shape


# In[49]:


X_train[0]


# In[50]:


plt.matshow(X_train[0])


# In[52]:


X_train.shape


# In[57]:


X_train = X_train/255
X_test = X_test/255


# In[58]:


X_test[0]


# In[66]:


X_train_flattened = X_train.reshape(len(X_train),28*28)
X_train_flattened.shape
X_test_flattened = X_test.reshape(len(X_test),28*28)
X_test_flattened.shape


# In[67]:


model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,),activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss = "sparse_categorical_crossentropy",
    metrics=["accuracy"])

model.fit(X_train_flattened, y_train, epochs = 10)


# In[69]:


model.evaluate(X_test_flattened, y_test)


# In[70]:


plt.matshow(X_test[0])


# In[71]:


y_predicted = model.predict(X_test_flattened)
y_predicted[0]


# In[72]:


np.argmax(y_predicted[0])


# In[73]:


model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,),activation="relu"),
    keras.layers.Dense(10, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss = "sparse_categorical_crossentropy",
    metrics=["accuracy"])

model.fit(X_train_flattened, y_train, epochs = 5)


# In[74]:


model.evaluate(X_test_flattened, y_test)


# In[75]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape= (28,28)),
    keras.layers.Dense(100, input_shape=(784,),activation="relu"),
    keras.layers.Dense(10, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss = "sparse_categorical_crossentropy",
    metrics=["accuracy"])

model.fit(X_train, y_train, epochs = 5)


# In[ ]:




