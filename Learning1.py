
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn


# In[4]:

import pandas as pd
counts = pd.Series([632,1638,569,115])


# In[5]:

counts.values


# In[6]:

counts.index


# In[13]:

bacteria = pd.Series([632,1638,569,115],
index = ['Firmicutes','Proteobacteria','Actinobacteria','Bacteroidetes'])


# In[14]:

bacteria


# In[15]:

bacteria['Firmicutes']


# In[17]:

bacteria[[name.endswith('bacteria') for name in bacteria.index]]


# In[18]:

[name.endswith('bacteria') for name in bacteria.index]


# In[19]:

bacteria.name = 'counts'


# In[20]:

bacteria.index.name = 'phylum'
bacteria


# In[ ]:



