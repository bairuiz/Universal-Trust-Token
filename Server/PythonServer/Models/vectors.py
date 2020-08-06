#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 14:44:32 2020

@author: chi
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import Word
import pickle

def clean_dataset(X):
    #remove digits 
    #remove words less than 3 characters 
    #remove punctuation

    X['clean_title'] = X['title'].str.replace('\d+', ' ') # for digits
    X['clean_title'] = X['clean_title'].str.replace(r'(\b\w{1,2}\b)', ' ') # for words less than 3 characters
    X['clean_title'] = X['clean_title'].str.replace('[^\w\s]', ' ') # for punctuation 
    
    X['clean_text'] = X['text'].str.replace('\d+', ' ') # for digits
    X['clean_text'] = X['clean_text'].str.replace(r'(\b\w{1,2}\b)', ' ') # for words less than 3 characters
    X['clean_text'] = X['clean_text'].str.replace('[^\w\s]', ' ') # for punctuation 
    #lemmatization 
    X['clean_title'] = X['clean_title'] .apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    X['clean_text'] = X['clean_text'] .apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    return X
    
def split_dataset(X, y):
    #Split to train and test dataset 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1000)

    #Reset all the index
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return (X_train, X_test, y_train, y_test)

def count_vectorizer_train(train_data, ngram):
    #train title data
    countVec_title = CountVectorizer(lowercase=True, stop_words='english', min_df=0.01, ngram_range=ngram, strip_accents='ascii')
    vector_train_title = countVec_title.fit_transform(train_data['clean_title'].values.astype('U'))
    tokens_title = countVec_title.get_feature_names()
    vectorized_train_title = pd.DataFrame(vector_train_title.toarray(), columns=tokens_title)

    #train text data
    countVec_text = CountVectorizer(lowercase=True, stop_words='english', min_df=0.01, ngram_range= ngram, strip_accents='ascii')
    vector_train_text = countVec_text.fit_transform(train_data['clean_text'].values.astype('U'))
    tokens_text = countVec_text.get_feature_names()
    vectorized_train_text = pd.DataFrame(vector_train_text.toarray(), columns=tokens_text)

    #combine train data and test data features
    vectorized_train = pd.concat([vectorized_train_title, vectorized_train_text], axis=1)
   
    return vectorized_train, countVec_title, countVec_text, tokens_title, tokens_text



def tfidf_vectorizer_train(train_data, ngram):
    #train title data
    tfidfVec_title = TfidfVectorizer(lowercase=True, stop_words='english', min_df=0.01, ngram_range= ngram, strip_accents='ascii')
    vector_train_title = tfidfVec_title.fit_transform(train_data['clean_title'].values.astype('U'))
    tokens_title = tfidfVec_title.get_feature_names()
    vectorized_train_title = pd.DataFrame(vector_train_title.toarray(), columns=tokens_title)
    
    #train text data
    tfidfVec_text = TfidfVectorizer(lowercase=True, stop_words='english', min_df=0.01, ngram_range= ngram, strip_accents='ascii')
    vector_train_text = tfidfVec_text.fit_transform(train_data['clean_text'].values.astype('U'))
    tokens_text = tfidfVec_text.get_feature_names()
    vectorized_train_text = pd.DataFrame(vector_train_text.toarray(), columns=tokens_text)   
    
    #combine train data features
    vectorized_train = pd.concat([vectorized_train_title, vectorized_train_text], axis=1)
    
    return vectorized_train, tfidfVec_title, tfidfVec_text, tokens_title, tokens_text
     
if __name__ == '__main__':
    data = pd.read_csv("newData_w_title.csv")
    
    #Divide data into feaures and labels
    X = data.loc[:,['title','text']]
    X = clean_dataset(X)
    y = pd.DataFrame(data['label'])
    
    #Split the data into train and test 
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    
    ngram = (1,3)
    
    #Tfidf Vectorize
    train_data, tfidfVec_title, tfidfVec_text, tokens_title, tokens_text = tfidf_vectorizer_train(X_train, ngram)
    pickle.dump(tfidfVec_title, open("tfidf_1-3_vector_title.sav", "wb"))
    pickle.dump(tfidfVec_text, open("tfidf_1-3_vector_text.sav", "wb"))
    
    #Count Vectorize
    train_data, countVec_title, countVec_text, tokens_title, tokens_text = count_vectorizer_train(X_train, ngram) 
    pickle.dump(countVec_title, open("cv_1-3_vector_title.sav", "wb"))
    pickle.dump(countVec_text, open("cv_1-3_vector_text.sav", "wb"))