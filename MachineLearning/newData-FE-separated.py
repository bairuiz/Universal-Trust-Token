#Libraries 
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.stem import WordNetLemmatizer,PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import Word
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
import pickle

def count_vectorizer(train_data, test_data, ngram):
    #train title data
    countVec_title = CountVectorizer(lowercase=True, stop_words='english', min_df =0.01, ngram_range= ngram, strip_accents='ascii')
    vector_train_title = countVec_title.fit_transform(train_data['clean_title'].values.astype('U'))
    tokens_title = countVec_title.get_feature_names()
    vectorized_train_title = pd.DataFrame(vector_train_title.toarray(), columns=tokens_title)
    
    #test title data - only transform
    vector_test_title = countVec_title.transform(test_data['clean_title'].values.astype('U'))
    vectorized_test_title = pd.DataFrame(vector_test_title.toarray(), columns=tokens_title)

    #train text data
    countVec_text = CountVectorizer(lowercase=True, stop_words='english', min_df =0.01, ngram_range= ngram, strip_accents='ascii')
    vector_train_text = countVec_text.fit_transform(train_data['clean_text'].values.astype('U'))
    tokens_text = countVec_text.get_feature_names()
    vectorized_train_text = pd.DataFrame(vector_train_text.toarray(), columns=tokens_text)
    
    #test text data - only transform
    vector_test_text = countVec_text.transform(test_data['clean_text'].values.astype('U'))
    vectorized_test_text = pd.DataFrame(vector_test_text.toarray(), columns=tokens_text)

    #combine train data and test data features
    vectorized_train = pd.concat([vectorized_train_title, vectorized_train_text], axis=1)
    vectorized_test = pd.concat([vectorized_test_title, vectorized_test_text], axis=1)
    return vectorized_train, vectorized_test

def tfidf_vectorizer(train_data, test_data, ngram):
    #train title data
    tfidfVec_title = TfidfVectorizer(lowercase=True, stop_words='english', min_df =0.01, ngram_range= ngram, strip_accents='ascii')
    vector_train_title = tfidfVec_title.fit_transform(train_data['clean_title'].values.astype('U'))
    tokens_title = tfidfVec_title.get_feature_names()
    vectorized_train_title = pd.DataFrame(vector_train_title.toarray(), columns=tokens_title)
    
    #test title data - only transform
    vector_test_title = tfidfVec_title.transform(test_data['clean_title'].values.astype('U'))
    vectorized_test_title = pd.DataFrame(vector_test_title.toarray(), columns=tokens_title)

    #train text data
    tfidfVec_text = TfidfVectorizer(lowercase=True, stop_words='english', min_df =0.01, ngram_range= ngram, strip_accents='ascii')
    vector_train_text = tfidfVec_text.fit_transform(train_data['clean_text'].values.astype('U'))
    tokens_text = tfidfVec_text.get_feature_names()
    vectorized_train_text = pd.DataFrame(vector_train_text.toarray(), columns=tokens_text)
    
    #test text data - only transform
    vector_test_text = tfidfVec_text.transform(test_data['clean_text'].values.astype('U'))
    vectorized_test_text = pd.DataFrame(vector_test_text.toarray(), columns=tokens_text)

    #combine train data and test data features
    vectorized_train = pd.concat([vectorized_train_title, vectorized_train_text], axis=1)
    vectorized_test = pd.concat([vectorized_test_title, vectorized_test_text], axis=1)
    return vectorized_train, vectorized_test

if __name__ == '__main__':
    # read the dataset
    data = pd.read_csv('newData.csv')
    
    X = pd.DataFrame(data[['title', 'text']])
    y = pd.DataFrame(data['label'])

    #######################################
    ############TEXT CLEANING##############
    #####APPLIED TO BOTH TRAIN AND TEST####
    #######################################
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


    #Count Vectorize - Unigram
    ngram = (1,1)
    train_data_count_uni, test_data_count_uni = count_vectorizer(X_train, X_test, ngram)
    
    print(train_data_count_uni)
    print(test_data_count_uni)
    
    #Count Vectorize - Bigram
    ngram = (1,2)
    train_data_count_bi, test_data_count_bi = count_vectorizer(X_train, X_test, ngram)
    
    print(train_data_count_bi)
    print(test_data_count_bi)

    #Count Vectorize - Trigram
    ngram = (1,3)
    train_data_count_tri, test_data_count_tri = count_vectorizer(X_train, X_test, ngram)
    
    print(train_data_count_tri)
    print(test_data_count_tri)
    
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
    
    with open('vectorizedData-separated.pkl', 'wb') as f:
        obj = (train_data_count_uni,
               test_data_count_uni,
               train_data_count_bi,
               test_data_count_bi,
               train_data_count_tri,
               test_data_count_tri,
               train_data_tfidf_uni,
               test_data_tfidf_uni,
               train_data_tfidf_bi,
               test_data_tfidf_bi,
               train_data_tfidf_tri,
               test_data_tfidf_tri,
               y_train,
               y_test)
        pickle.dump(obj, f)

