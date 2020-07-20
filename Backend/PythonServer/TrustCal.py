import pickle
import pandas as pd
from textblob import Word


def loadVectors():
    #loading CountVec object
    cv_vector_title = pickle.load(open('cv_1-3_vector_title.sav','rb'))
    cv_vector_text = pickle.load(open('cv_1-3_vector_text.sav','rb'))
    
    #loading TF-IDF object
    tfidf_vector_title = pickle.load(open('tfidf_1-3_vector_title.sav','rb'))
    tfidf_vector_text = pickle.load(open('tfidf_1-3_vector_text.sav','rb'))
    
    return cv_vector_title, cv_vector_text, tfidf_vector_title, tfidf_vector_text

def loadModels():
    svm = pickle.load(open('SVM_tfidf_1-3.sav','rb'))
    rf = pickle.load(open('RandomForest_model_countvec_trigram.sav','rb'))
    lr = pickle.load(open('LR_countvec_1-3.sav','rb'))
    return svm,rf,lr

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

def makeCV_vector(df,cv_vector_title,cv_vector_text):
    vector_title = cv_vector_title.transform(df['clean_title'].values.astype('U'))
    vectorized_title = pd.DataFrame(vector_title.toarray(), columns=cv_vector_title.get_feature_names())
    
    vector_text = cv_vector_text.transform(df['clean_text'].values.astype('U'))
    vectorized_text = pd.DataFrame(vector_text.toarray(), columns=cv_vector_text.get_feature_names())
    
    #combine test title and text
    vectorized_cv = pd.concat([vectorized_title, vectorized_text], axis=1)
    return vectorized_cv
    
def makeTFIDF_vector(df,tfidf_vector_title,tfidf_vector_text):
    vector_title = tfidf_vector_title.transform(df['clean_title'].values.astype('U'))
    vectorized_title = pd.DataFrame(vector_title.toarray(), columns=tfidf_vector_title.get_feature_names())

    vector_text = tfidf_vector_text.transform(df['clean_text'].values.astype('U'))
    vectorized_text = pd.DataFrame(vector_text.toarray(), columns=tfidf_vector_text.get_feature_names())
    
    vectorized_tfidf = pd.concat([vectorized_title, vectorized_text], axis=1)
    return vectorized_tfidf

def calculate(df,cv_vector_title, cv_vector_text, tfidf_vector_title, tfidf_vector_text, svm, rf, lr):
    df = clean_dataset(df)
    vectorized_cv = makeCV_vector(df,cv_vector_title,cv_vector_text)
    vectorized_tfidf = makeTFIDF_vector(df,tfidf_vector_title,tfidf_vector_text)
    prediction_svm = svm.predict_proba(vectorized_tfidf)
    prediction_rf = rf.predict_proba(vectorized_cv)
    prediction_lr = lr.predict_proba(vectorized_cv)
    percentage = (prediction_svm[0][1] + prediction_rf[0][1] + prediction_lr[0][1])/3 * 100
    return str(int(round(percentage,0)))