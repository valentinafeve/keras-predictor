#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[2]:


get_ipython().system('pip install -q -U tensorflow')


# In[3]:


import itertools
import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
layers = keras.layers


# In[ ]:


get_ipython().system('wget https://storage.googleapis.com/sara-cloud-ml/wine_data.csv')


# In[4]:


path = "wine_data.csv"


# In[5]:


data = pd.read_csv(path)
data = data.sample(frac=1)
data.head()


# In[6]:


data = data[pd.notnull(data['country'])]
data = data[pd.notnull(data['price'])]
data = data.drop(data.columns[0], axis=1)
variety_threshold = 500
value_counts = data['variety'].value_counts()
to_remove = value_counts[value_counts <= variety_threshold].index
data.replace(to_remove , np.nan, inplace=True)
data = data[pd.notnull(data['variety'])]


# In[7]:


train_size= int(len(data) * .8)
print("Train size: %d" % train_size)
print("Test size: %d" % (len(data) - train_size))


# In[8]:


# Train inputs
description_train = data['description'][:train_size]
variety_train = data['variety'][:train_size]

# Train labels
labels_train = data['price'][:train_size]

# Test inputs
description_test = data['description'][train_size:]
variety_test = data['variety'][train_size:]

# Test labels
labels_test = data['price'][train_size:]


# In[9]:


# Create a tokenizer
vocab_size = 5000
tokenize = keras.preprocessing.text.Tokenizer(num_words=vocab_size, char_level=False)
tokenize.fit_on_texts(description_train)


# In[10]:


description_bow_train = tokenize.texts_to_matrix(description_train)
description_bow_test = tokenize.texts_to_matrix(description_test)


# In[11]:


encoder = LabelEncoder()
encoder.fit(variety_train)
variety_train = encoder.transform(variety_train)
variety_test = encoder.transform(variety_test)
num_classes = np.max(variety_train) + 1

#To one-hot
variety_train = keras.utils.to_categorical(variety_train, num_classes)
variety_test = keras.utils.to_categorical(variety_test, num_classes)


# In[12]:


bow_inputs = layers.Input(shape=(vocab_size,))
variety_inputs = layers.Input(shape=(num_classes,))
merged_layer= layers.concatenate([bow_inputs, variety_inputs])
merget_layer = layers.Dense(256, activation="relu")(merged_layer)
predictions = layers.Dense(1)(merged_layer)
wide_model = keras.Model(inputs=[bow_inputs, variety_inputs], outputs=predictions)


# In[13]:


wide_model.compile(loss="mse", optimizer='adam', metrics=['accuracy'])
print(wide_model.summary())


# In[14]:


train_embed = tokenize.texts_to_sequences(description_train)
test_embed = tokenize.texts_to_sequences(description_test)

max_seq_length = 170
train_embed = keras.preprocessing.sequence.pad_sequences(train_embed, maxlen=max_seq_length, padding = "post")
test_embed = keras.preprocessing.sequence.pad_sequences(test_embed, maxlen=max_seq_length, padding = "post")


# In[15]:


deep_inputs = layers.Input(shape=(max_seq_length,))
embedding = layers.Embedding(vocab_size, 8, input_length=max_seq_length)(deep_inputs)
embedding = layers.Flatten()(embedding)
embed_out = layers.Dense(1)(embedding)
deep_model = keras.Model(inputs=deep_inputs, outputs=embed_out)
print(deep_model.summary())


# In[16]:


deep_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])


# In[17]:


merged_out = layers.concatenate([wide_model.output, deep_model.output])
merged_out = layers.Dense(1)(merged_out)
combined_model = keras.Model(wide_model.input + [deep_model.input], merged_out)
print(combined_model.summary())


# In[18]:


combined_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])


# In[22]:


combined_model.fit([description_bow_train, variety_train] + [ train_embed ], labels_train, epochs=50, batch_size=128)
combined_model.evaluate([description_bow_test, variety_test] + [ test_embed ], labels_test, batch_size=128)


# In[23]:


predictions = combined_model.predict([ description_bow_test, variety_test] + [ test_embed])


# In[24]:


num_predictions = 40
diff = 0
for i in range(num_predictions):
  val = predictions[i]
  print(description_test.iloc[i])
  print('Predicted: ', val[0], 'Actual: ', labels_test.iloc[i], '\n')
  diff += abs(val[0] - labels_test.iloc[i])


# In[ ]:




