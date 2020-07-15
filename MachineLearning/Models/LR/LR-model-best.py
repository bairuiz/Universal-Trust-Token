#####################################
######## Logistic Regression ########
#####################################

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics

def evalMetrics(model, train_data, test_data, y_train, y_test):
    print('\n======= Train Data =======')
    p_pred_train = model.predict_proba(train_data)
    y_pred_train = model.predict(train_data)
    score_train = model.score(train_data, y_train)
    conf_m_train = confusion_matrix(y_train, y_pred_train)
    report_train = classification_report(y_train, y_pred_train)

    Accuracy_train = metrics.accuracy_score(y_train, y_pred_train)
    Precision_train = metrics.precision_score(y_train, y_pred_train, average='macro')
    F1_Score_train = metrics.f1_score(y_train, y_pred_train, average='macro')
    Recall_train = metrics.recall_score(y_train, y_pred_train, average='macro')

    
    print('p_pred:', p_pred_train, sep='\n', end='\n\n')
    print('y_pred:', y_pred_train, end='\n\n')
    print('score_:', score_train, end='\n\n')
    print('conf_m:', conf_m_train, sep='\n', end='\n\n')
    print('report:', report_train, sep='\n')
    
    print('Accuracy:', round(Accuracy_train * 100, 2), sep=' ')
    print('Precision:', round(Precision_train * 100, 2), sep=' ')
    print('F1_Score:', round(F1_Score_train * 100, 2), sep=' ')
    print('Recall:', round(Recall_train * 100, 2), sep=' ')


    print('\n======= Test Data =======')
    p_pred_test = model.predict_proba(test_data)
    y_pred_test = model.predict(test_data)
    score_test = model.score(test_data, y_test)
    conf_m_test = confusion_matrix(y_test, y_pred_test)
    report_test = classification_report(y_test, y_pred_test)

    Accuracy_test = metrics.accuracy_score(y_test, y_pred_test)
    Precision_test = metrics.precision_score(y_test, y_pred_test, average='macro')
    F1_Score_test = metrics.f1_score(y_test, y_pred_test, average='macro')
    Recall_test = metrics.recall_score(y_test, y_pred_test, average='macro')

    
    print('p_pred:', p_pred_test, sep='\n', end='\n\n')
    print('y_pred:', y_pred_test, end='\n\n')
    print('score_:', score_test, end='\n\n')
    print('conf_m:', conf_m_test, sep='\n', end='\n\n')
    print('report:', report_test, sep='\n')
    
    print('Accuracy:', round(Accuracy_test * 100, 2), sep=' ')
    print('Precision:', round(Precision_test * 100, 2), sep=' ')
    print('F1_Score:', round(F1_Score_test * 100, 2), sep=' ')
    print('Recall:', round(Recall_test * 100, 2), sep=' ')


if __name__ == '__main__':
    # read in vectorized data
    with open('vectorizedData-bestLR.pkl', 'rb') as f:
        (train_data_count_tri,
         test_data_count_tri,
         tokens_title_count_tri,
         tokens_text_count_tri,
         y_train,
         y_test) = pickle.load(f)
    
    # Logistic Regression for Count Vectorized data(Trigram)
    tri_count_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
    tri_count_model.fit(train_data_count_tri, y_train)
    print('\n------- Count(Trigram) -------')
    evalMetrics(tri_count_model, train_data_count_tri, test_data_count_tri, y_train, y_test)
    
    # pickle model, tokens_title and tokens_text
    pickle.dump(tri_count_model,open('LR_count_tri.sav','wb'))
    pickle.dump(tokens_title_count_tri,open('LR_count_tri_title.sav','wb'))
    pickle.dump(tokens_text_count_tri,open('LR_count_tri_text.sav','wb'))
    
   
