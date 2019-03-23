import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np




def read_csv(file_path):
    df = pd.read_csv(file_path)
    df = df[['Short Description', 'Incident Type', 'Root Cause', 'Design Related Potential']]
    train_data, train_label = [], []
    test_data, test_label = [], []
    unseen_data, unseen_label = [], []
    
    for index, row in df.iterrows():
        string = (str(row['Short Description']) + " " + 
                  str(row['Incident Type']) + " " + 
                  str(row['Root Cause']))
        label = str(row['Design Related Potential'])

        if label == 'y' or label == 'n':
            train_data.append(string)
            train_label.append(label)

        unseen_data.append(string)
        unseen_label.append(label)
    # split train + test data
    TRAIN_TEST = 60
    test_data = train_data[TRAIN_TEST+1:]
    test_label = train_label[TRAIN_TEST+1:]
    train_data = train_data[:TRAIN_TEST]
    train_label= train_label[:TRAIN_TEST]

    return train_data, train_label, test_data, test_label, unseen_data, unseen_label


if __name__ == "__main__":
    path = "incidents.csv"
    train_data, train_label, test_data, test_label, unseen_data, unseen_label = read_csv(path)
    
    MODEL = "NB"
    if MODEL == "NB":
        text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB())])
        parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                      'tfidf__use_idf': (True, False),
                      'clf__alpha': (1e-2, 1e-3)}
    elif MODEL == "SVM":
        text_clf = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                                   alpha=1e-3,  
                                                   random_state=42))
                             ])
        parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                      'tfidf__use_idf': (True, False),
                      'clf__alpha': (1e-2, 1e-3)}

    text_clf = text_clf.fit(train_data, train_label)
    predicted = text_clf.predict(test_data)
    print("non grid search", np.mean(predicted == test_label))
    

    
    
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(train_data, train_label)
    predicted = gs_clf.predict(test_data)
    print("grid search", np.mean(predicted == test_label))
    #print(gs_clf.best_score_)
    #print(gs_clf.best_params_)


    # Model validation
    print(np.mean(predicted == test_label))

    """
    #for i in range(len(test_data)):
        #print(test_data[i], '|', predicted[i], '|', test_label[i])

    # Testing against unseen 
    unseen_predicted = text_clf.predict(unseen_data)
    #print(np.mean(predicted == test_label))
    no, yes = 0, 0
    for i in range(len(unseen_data)):
        #print(unseen_data[i], '|', unseen_predicted[i])
        if unseen_predicted[i] == 'y':
            yes += 1
        else:
            no += 1

    print("Yes:", yes, "No:" , no)

"""
