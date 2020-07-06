#Libraries 
import pandas as pd
from sklearn.model_selection import train_test_split
import string 
import re
import nltk
from nltk.stem import WordNetLemmatizer,PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import Word
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

def count_vectorizer(train_data, test_data, ngram):
    #train data
    countVec = CountVectorizer(lowercase=True, stop_words='english', min_df =0.01, ngram_range= ngram, strip_accents='ascii')
    vector_train = countVec.fit_transform(train_data['clean_text'].values.astype('U'))
    tokens = countVec.get_feature_names()
    vectorized_train = pd.DataFrame(vector_train.toarray(), columns=tokens)
    
    #test data - only transform
    vector_test = countVec.transform(test_data['clean_text'].values.astype('U'))
    vectorized_test = pd.DataFrame(vector_test.toarray(), columns=tokens)
    return vectorized_train, vectorized_test

def tfidf_vectorizer(train_data, test_data, ngram):
    #train data
    tfidfVec = TfidfVectorizer(lowercase=True, stop_words='english', min_df =0.01, ngram_range= ngram, strip_accents='ascii')
    vector_train = tfidfVec.fit_transform(train_data['clean_text'].values.astype('U'))
    tokens = tfidfVec.get_feature_names()
    vectorized_train = pd.DataFrame(vector_train.toarray(), columns=tokens)
    
    #test data - only transform
    vector_test = tfidfVec.transform(test_data['clean_text'].values.astype('U'))
    vectorized_test = pd.DataFrame(vector_test.toarray(), columns=tokens)
    return vectorized_train, vectorized_test



if __name__ == '__main__':
    # read the dataset
    data = pd.read_csv('newData.csv', index_col=0)
    
    X = pd.DataFrame(data['combined_text'])
    y = pd.DataFrame(data['label'])

    #######################################
    ############TEXT CLEANING##############
    #####APPLIED TO BOTH TRAIN AND TEST####
    #######################################
    #remove digits 
    #remove words less than 3 characters 
    #remove punctuation
    
    X['clean_text'] = X['combined_text'].str.replace('\d+', ' ') # for digits
    X['clean_text'] = X['clean_text'].str.replace(r'(\b\w{1,2}\b)', ' ') # for words less than 3 characters
    X['clean_text'] = X['clean_text'].str.replace('[^\w\s]', ' ') # for punctuation 
    
    #lemmatization 
    X['clean_text'] = X['clean_text'] .apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    
    
    #Split to train and test dataset 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1000)
    
    #Reset all the index
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    print(X_train.head())
    print(y_train['label'].value_counts())
    print(y_test['label'].value_counts())


    #Count Vectorize
    ngram = (1,1)
    train_data, test_data = count_vectorizer(X_train, X_test, ngram)
    
    print(train_data)
    print(test_data)
    
    #Tfidf Vectorize - Unigram
    ngram = (1,1)
    train_data_tfidf_uni, test_data_tfidf_uni = tfidf_vectorizer(X_train, X_test, ngram)
    
    print(train_data_tfidf_uni)
    print(test_data_tfidf_uni)
    
    #Tfidf Vectorize - Bigram
    ngram = (1,2)
    train_data_tfidf_bi, test_data_tfidf_bi = tfidf_vectorizer(X_train, X_test, ngram)
    
    print(train_data_tfidf_bi)
    print(test_data_tfidf_bi)
    
    #Tfidf Vectorize - Trigram
    ngram = (1,3)
    train_data_tfidf_tri, test_data_tfidf_tri = tfidf_vectorizer(X_train, X_test, ngram)
    
    print(train_data_tfidf_tri)
    print(test_data_tfidf_tri)

    #####################################
    ######## Logistic Regression ########
    #####################################
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import cross_val_score
    
    # Logistic Regression for Count Vectorized data
    count_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.2)
    count_model.fit(train_data, y_train)
    
    p_pred_count = count_model.predict_proba(test_data)
    y_pred_count = count_model.predict(test_data)
    score_count = count_model.score(test_data, y_test)
#    cv_score_count = cross_val_score(count_model, train_data, y_train, cv=10)
    conf_m_count = confusion_matrix(y_test, y_pred_count)
    report_count = classification_report(y_test, y_pred_count)

    print('\n------- Count Vectorize -------')
    print('p_pred:', p_pred_count, sep='\n', end='\n\n')
    print('y_pred:', y_pred_count, end='\n\n')
    print('score_:', score_count, end='\n\n')
#    print('cv_score_:', cv_score_count, end='\n\n')
    print('conf_m:', conf_m_count, sep='\n', end='\n\n')
    print('report:', report_count, sep='\n')

    # Logistic Regression for TF-IDF(Unigram)
    uni_model = LogisticRegression(penalty='l1', solver='liblinear', C=1)
    uni_model.fit(train_data_tfidf_uni, y_train)
    
    p_pred_uni = uni_model.predict_proba(test_data_tfidf_uni)
    y_pred_uni = uni_model.predict(test_data_tfidf_uni)
    score_uni = uni_model.score(test_data_tfidf_uni, y_test)
#    cv_score_uni = cross_val_score(uni_model, train_data_tfidf_uni, y_train, cv=10)
    conf_m_uni = confusion_matrix(y_test, y_pred_uni)
    report_uni = classification_report(y_test, y_pred_uni)
    
    print('\n------- TF-IDF(Unigram) -------')
    print('p_pred:', p_pred_uni, sep='\n', end='\n\n')
    print('y_pred:', y_pred_uni, end='\n\n')
    print('score_:', score_uni, end='\n\n')
#    print('cv_score_:', cv_score_uni, end='\n\n')
    print('conf_m:', conf_m_uni, sep='\n', end='\n\n')
    print('report:', report_uni, sep='\n')

    # Logistic Regression for TF-IDF(Bigram)
    bi_model = LogisticRegression(penalty='l1', solver='liblinear', C=1)
    bi_model.fit(train_data_tfidf_bi, y_train)
    
    p_pred_bi = bi_model.predict_proba(test_data_tfidf_bi)
    y_pred_bi = bi_model.predict(test_data_tfidf_bi)
    score_bi = bi_model.score(test_data_tfidf_bi, y_test)
#    cv_score_bi = cross_val_score(bi_model, train_data_tfidf_bi, y_train, cv=10)
    conf_m_bi = confusion_matrix(y_test, y_pred_bi)
    report_bi = classification_report(y_test, y_pred_bi)
    
    print('\n------- TF-IDF(Bigram) -------')
    print('p_pred:', p_pred_bi, sep='\n', end='\n\n')
    print('y_pred:', y_pred_bi, end='\n\n')
    print('score_:', score_bi, end='\n\n')
#    print('cv_score_:', cv_score_bi, end='\n\n')
    print('conf_m:', conf_m_bi, sep='\n', end='\n\n')
    print('report:', report_bi, sep='\n')
    
    # Logistic Regression for TF-IDF(Trigram)
    tri_model = LogisticRegression(penalty='l1', solver='liblinear', C=1)
    tri_model.fit(train_data_tfidf_tri, y_train)
    
    p_pred_tri = tri_model.predict_proba(test_data_tfidf_tri)
    y_pred_tri = tri_model.predict(test_data_tfidf_tri)
    score_tri = tri_model.score(test_data_tfidf_tri, y_test)
#    cv_score_tri = cross_val_score(tri_model, train_data_tfidf_tri, y_train, cv=10)
    conf_m_tri = confusion_matrix(y_test, y_pred_tri)
    report_tri = classification_report(y_test, y_pred_tri)
    
    print('\n------- TF-IDF(Trigram) -------')
    print('p_pred:', p_pred_tri, sep='\n', end='\n\n')
    print('y_pred:', y_pred_tri, end='\n\n')
    print('score_:', score_tri, end='\n\n')
#    print('cv_score_:', cv_score_tri, end='\n\n')
    print('conf_m:', conf_m_tri, sep='\n', end='\n\n')
    print('report:', report_tri, sep='\n')
    
    
