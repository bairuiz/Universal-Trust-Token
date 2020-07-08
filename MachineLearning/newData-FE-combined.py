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
    
    with open('vectorizedData-combined.pkl', 'wb') as f:
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

