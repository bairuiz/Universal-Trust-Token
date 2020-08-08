#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Libraries 
import pandas as pd
import numpy as np
np.set_printoptions(formatter={'float_kind':'{:f}'.format})
from sklearn.model_selection import train_test_split
import string 
import re
import nltk
from nltk.stem import WordNetLemmatizer,PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import Word
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.neural_network import MLPClassifier
import pickle


# In[2]:


def count_vectorizer(train_data, test_data, ngram):
    #train title data
    countVec_title = CountVectorizer(lowercase=True, stop_words='english', min_df =0.01, ngram_range= ngram, strip_accents='ascii')
    #vector_train_title = countVec_title.fit_transform(train_data['clean_title'].values.astype('U'))
    vector_train_title = countVec_title.fit_transform(train_data['clean_title'])
    tokens_title = countVec_title.get_feature_names()
    vectorized_train_title = pd.DataFrame(vector_train_title.toarray(), columns=tokens_title)
    
    #test title data - only transform
#    vector_test_title = countVec_title.transform(test_data['clean_title'].values.astype('U'))
    vector_test_title = countVec_title.transform(test_data['clean_title'])
    vectorized_test_title = pd.DataFrame(vector_test_title.toarray(), columns=tokens_title)

    #train text data
    countVec_text = CountVectorizer(lowercase=True, stop_words='english', min_df =0.01, ngram_range= ngram, strip_accents='ascii')
    #vector_train_text = countVec_text.fit_transform(train_data['clean_text'].values.astype('U'))
    vector_train_text = countVec_text.fit_transform(train_data['clean_text'])
    tokens_text = countVec_text.get_feature_names()
    vectorized_train_text = pd.DataFrame(vector_train_text.toarray(), columns=tokens_text)
    
    #test text data - only transform
    #vector_test_text = countVec_text.transform(test_data['clean_text'].values.astype('U'))
    vector_test_text = countVec_text.transform(test_data['clean_text'])
    vectorized_test_text = pd.DataFrame(vector_test_text.toarray(), columns=tokens_text)

    #combine train data and test data features
    vectorized_train = pd.concat([vectorized_train_title, vectorized_train_text], axis=1)
    vectorized_test = pd.concat([vectorized_test_title, vectorized_test_text], axis=1)
    return vectorized_train, vectorized_test

def tfidf_vectorizer(train_data, test_data, ngram):
    #train title data
    tfidfVec_title = TfidfVectorizer(lowercase=True, stop_words='english', min_df =0.01, ngram_range= ngram, strip_accents='ascii')
#    vector_train_title = tfidfVec_title.fit_transform(train_data['clean_title'].values.astype('U'))
    vector_train_title = tfidfVec_title.fit_transform(train_data['clean_title'])
    tokens_title = tfidfVec_title.get_feature_names()
    vectorized_train_title = pd.DataFrame(vector_train_title.toarray(), columns=tokens_title)
    
    #test title data - only transform
#    vector_test_title = tfidfVec_title.transform(test_data['clean_title'].values.astype('U'))
    vector_test_title = tfidfVec_title.transform(test_data['clean_title'])
    vectorized_test_title = pd.DataFrame(vector_test_title.toarray(), columns=tokens_title)

    #train text data
    tfidfVec_text = TfidfVectorizer(lowercase=True, stop_words='english', min_df =0.01, ngram_range= ngram, strip_accents='ascii')
#    vector_train_text = tfidfVec_text.fit_transform(train_data['clean_text'].values.astype('U'))
    vector_train_text = tfidfVec_text.fit_transform(train_data['clean_text'])
    tokens_text = tfidfVec_text.get_feature_names()
    vectorized_train_text = pd.DataFrame(vector_train_text.toarray(), columns=tokens_text)
    
    #test text data - only transform
#    vector_test_text = tfidfVec_text.transform(test_data['clean_text'].values.astype('U'))
    vector_test_text = tfidfVec_text.transform(test_data['clean_text'])
    vectorized_test_text = pd.DataFrame(vector_test_text.toarray(), columns=tokens_text)

    #combine train data and test data features
    vectorized_train = pd.concat([vectorized_train_title, vectorized_train_text], axis=1)
    vectorized_test = pd.concat([vectorized_test_title, vectorized_test_text], axis=1)
    return vectorized_train, vectorized_test


# In[3]:


#New Dataset to be tried
data = pd.read_csv("newData_w_title.csv")

#data.shape
#data.head()


X = pd.DataFrame(data[['title', 'text']])
y = pd.DataFrame(data['label'])


X['clean_title'] = X['title'].str.replace('\d+', ' ') # for digits
X['clean_title'] = X['clean_title'].str.replace(r'(\b\w{1,2}\b)', ' ') # for words less than 3 characters
X['clean_title'] = X['clean_title'].str.replace('[^\w\s]', ' ') # for punctuation 
    
X['clean_text'] = X['text'].str.replace('\d+', ' ') # for digits
X['clean_text'] = X['clean_text'].str.replace(r'(\b\w{1,2}\b)', ' ') # for words less than 3 characters
X['clean_text'] = X['clean_text'].str.replace('[^\w\s]', ' ') # for punctuation 
    
#lemmatization 
X['clean_title'] = X['clean_title'] .apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
X['clean_text'] = X['clean_text'] .apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    
    
#Split to train and test dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1000)

#Reset all the index
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


# In[4]:


#Count Vectorize
#ngram = (1,1)
#train_data_uni, test_data_uni = count_vectorizer(X_train, X_test, ngram)

#Count Vectorize
#ngram = (1,2)
#train_data_bi, test_data_bi = count_vectorizer(X_train, X_test, ngram)

#Count Vectorize
ngram = (1,3)
train_data_tri, test_data_tri = count_vectorizer(X_train, X_test, ngram)

#Tfidf Vectorize
#ngram = (1,1)
#train_data_tfidf_uni, test_data_tfidf_uni = tfidf_vectorizer(X_train, X_test, ngram)

#Tfidf Vectorize
#ngram = (1,2)
#train_data_tfidf_bi, test_data_tfidf_bi = tfidf_vectorizer(X_train, X_test, ngram)

#Tfidf Vectorize
ngram = (1,3)
train_data_tfidf_tri, test_data_tfidf_tri = tfidf_vectorizer(X_train, X_test, ngram)


# In[5]:


with open('vectorizedData-MLP.pkl', 'wb') as f:
        obj = (train_data_tri,
               test_data_tri,
               train_data_tfidf_tri,
               test_data_tfidf_tri,
               y_train,
               y_test)
        pickle.dump(obj, f)


# In[6]:


vector_data= pd.read_pickle("vectorizedData-MLP.pkl")


# In[7]:

# Please check the index for the arrays if not using "vectorizedData-MLP.pkl"
#Trigram - countvector best model 
train_data = vector_data[0] 
test_data = vector_data[1]

#Trigram - tfidf best model
train_data_tfidf = vector_data[2]
test_data_tfidf = vector_data[3]


# In[8]:


y_train = vector_data[4]
y_test = vector_data[5]


# In[9]:


def evaluation_matrix(y_test, y_pred):
    #y_pred = y_pred.astype(int)
    #y_test = y_test.astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f_score, support = precision_recall_fscore_support(y_test, y_pred, average='macro')
    return accuracy, precision, recall, f_score 


# In[11]:


clf_countvec = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(30,30,30), random_state=1,max_iter=400)

clf_countvec.fit(train_data_tri, y_train['label'])

y_pred_countvec_train = clf_countvec.predict(train_data_tri)

accuracy_countvec_train, precision_countvec_train, recall_countvec_train, f_score_countvec_train = evaluation_matrix(y_train, y_pred_countvec_train)

print('MLP Classifier CountVec Tri Train - Hidden Layers different')
print('accuracy= ', accuracy_countvec_train)
print('precision= ', precision_countvec_train)
print('recall= ', recall_countvec_train)
print('f_score= ', f_score_countvec_train)
print('\n')


y_pred_countvec_test = clf_countvec.predict(test_data_tri)

accuracy_countvec_test, precision_countvec_test, recall_countvec_test, f_score_countvec_test = evaluation_matrix(y_test, y_pred_countvec_test)

print('MLP Classifier CountVec Tri Test - Hidden Layers different')
print('accuracy= ', accuracy_countvec_test)
print('precision= ', precision_countvec_test)
print('recall= ', recall_countvec_test)
print('f_score= ', f_score_countvec_test)
print('\n')


# In[25]:


clf_tfidf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(30,30,30), random_state=1,max_iter=400)

clf_tfidf.fit(train_data_tfidf_tri, y_train['label'])

y_pred_tfidf_train = clf_tfidf.predict(train_data_tfidf_tri)

accuracy_tfidf_train, precision_tfidf_train, recall_tfidf_train, f_score_tfidf_train = evaluation_matrix(y_train, y_pred_tfidf_train)

print('MLP Classifier TFIDF Tri Train - Hidden Layers different')
print('accuracy= ', accuracy_tfidf_train)
print('precision= ', precision_tfidf_train)
print('recall= ', recall_tfidf_train)
print('f_score= ', f_score_tfidf_train)
print('\n')


y_pred_tfidf_test = clf_tfidf.predict(test_data_tfidf_tri)

accuracy_tfidf_test, precision_tfidf_test, recall_tfidf_test, f_score_tfidf_test = evaluation_matrix(y_test, y_pred_tfidf_test)

print('MLP Classifier TFIDF Tri Test - Hidden Layers different')
print('accuracy= ', accuracy_tfidf_test)
print('precision= ', precision_tfidf_test)
print('recall= ', recall_tfidf_test)
print('f_score= ', f_score_tfidf_test)
print('\n')


# In[26]:


# save the model to disk
filename = 'finalized_model_countvec_tri_mlp.sav'
pickle.dump(clf_countvec, open(filename, 'wb'))


# In[27]:


# save the model to disk
filename_tfidf = 'finalized_model_tfidf_tri_mlp.sav'
pickle.dump(clf_tfidf, open(filename_tfidf, 'wb'))


# In[28]:



# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
y_pred = loaded_model.predict(test_data)
accuracy, precision, recall, f_score = evaluation_matrix(y_test, y_pred)

print('CountVec Test')
print('accuracy= ', accuracy)
print('precision= ', precision)
print('recall= ', recall)
print('f_score= ', f_score)


# In[29]:


# some time later...
 
# load the model from disk
loaded_model_tfidf = pickle.load(open(filename_tfidf, 'rb'))
y_pred_tfidf = loaded_model_tfidf.predict(test_data_tfidf)
accuracy_tfidf, precision_tfidf, recall_tfidf, f_score_tfidf = evaluation_matrix(y_test, y_pred_tfidf)

print('Tfidf test')
print('accuracy= ', accuracy_tfidf)
print('precision= ', precision_tfidf)
print('recall= ', recall_tfidf)
print('f_score= ', f_score_tfidf)


# In[33]:


y_pred = loaded_model.predict_proba(test_data)
y_pred


# In[ ]:





# In[ ]:




