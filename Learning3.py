
# coding: utf-8

# [Sebastian Raschka](http://sebastianraschka.com), 2015
# 
# https://github.com/rasbt/python-machine-learning-book

# # Python Machine Learning - Code Examples

# # Chapter 8 - Applying Machine Learning To Sentiment Analysis

# Note that the optional watermark extension is a small IPython notebook plugin that I developed to make the code reproducible. You can just skip the following line(s).

# In[ ]:




# In[1]:

# to install watermark just uncomment the following line:
#%install_ext https://raw.githubusercontent.com/rasbt/watermark/master/watermark.py


# <br>
# <br>

# # Read data

# In[2]:

import pyprind
import pandas as pd
import os
os.getcwd()
df = pd.read_csv('table1_python_test.csv')
df.head()


# Shuffling the DataFrame:

# In[3]:

import numpy as np
np.random.seed(0)

df = df.reindex(np.random.permutation(df.index))
df.to_csv('./test_shuffle.csv', index=False)


# In[ ]:




# Optional: Saving the assembled data as CSV file:

# In[ ]:




# In[4]:


df.head(3)


# <br>

# <br>

# ## Cleaning text data

# In[5]:

df.loc[0, 'narrative'][-50:]


# In[6]:

import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +            ' '.join(emoticons).replace('-', '')
    return text


# In[7]:

preprocessor(df.loc[0, 'narrative'][-50:])


# In[8]:

preprocessor("</a>This :) is :( a test :-)!")


# In[9]:

df['narrative'] = df['narrative'].apply(preprocessor)


# In[10]:

df.head(10)


# <br>

# ## Processing documents into tokens

# In[11]:

from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

def tokenizer(text):
    return text.split()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


# In[12]:

import nltk
nltk.download('stopwords')


# In[13]:

from nltk.corpus import stopwords
stop = stopwords.words('english')


# <br>
# <br>

# # Training a logistic regression model for document classification

# Strip HTML and punctuation to speed up the GridSearch later:

# In[14]:

X_train = df.loc[:700, 'narrative'].values
y_train = df.loc[:700, 'response'].values
X_test = df.loc[300:, 'narrative'].values
y_test = df.loc[300:, 'response'].values
X_train[1]


# In[ ]:

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents=None, 
                        lowercase=False, 
                        preprocessor=None)

param_grid = [{'vect__ngram_range': [(1,1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
             {'vect__ngram_range': [(1,1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
             ]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, 
                           scoring='accuracy',
                           cv=5, verbose=1,
                           n_jobs=-1)


# In[ ]:

gs_lr_tfidf.fit(X_train, y_train)


# In[ ]:

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')


# In[ ]:

print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)


# In[ ]:

clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))


# In[ ]:

from sklearn.metrics import accuracy_score,classification_report, confusion_matrix


# <hr>
# <hr>

# ####  Start comment:
#     
# Please note that `gs_lr_tfidf.best_score_` is the average k-fold cross-validation score. I.e., if we have a `GridSearchCV` object with 5-fold cross-validation (like the one above), the `best_score_` attribute returns the average score over the 5-folds of the best model. To illustrate this with an example:

# In[ ]:

from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np

np.random.seed(0)
np.set_printoptions(precision=6)
y = [np.random.randint(3) for i in range(25)]
X = (y + np.random.randn(25)).reshape(-1, 1)

cv5_idx = list(StratifiedKFold(y, n_folds=5, shuffle=False, random_state=0))
cross_val_score(LogisticRegression(random_state=123), X, y, cv=cv5_idx)


# By executing the code above, we created a simple data set of random integers that shall represent our class labels. Next, we fed the indices of 5 cross-validation folds (`cv3_idx`) to the `cross_val_score` scorer, which returned 5 accuracy scores -- these are the 5 accuracy values for the 5 test folds.  
# 
# Next, let us use the `GridSearchCV` object and feed it the same 5 cross-validation sets (via the pre-generated `cv3_idx` indices):

# In[ ]:

from sklearn.grid_search import GridSearchCV
gs = GridSearchCV(LogisticRegression(), {}, cv=cv5_idx, verbose=3).fit(X, y) 


# As we can see, the scores for the 5 folds are exactly the same as the ones from `cross_val_score` earlier.

# Now, the best_score_ attribute of the `GridSearchCV` object, which becomes available after `fit`ting, returns the average accuracy score of the best model:

# In[ ]:

gs.best_score_


# As we can see, the result above is consistent with the average score computed the `cross_val_score`.

# In[ ]:

cross_val_score(LogisticRegression(), X, y, cv=cv5_idx).mean()


# <br>
# <br>
