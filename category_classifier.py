import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split

class Data:
    def __init__(self):
        self.data = []
        self.label = []
        self.date = []
        self.id = []
            
def read_csv(file_path, scramble=False):
    df = pd.read_csv(file_path);
    if (scramble):
        df = df.sample(frac=1)
    train = Data()
    test = Data()
    unseen = Data()
    classified = Data()

    for index, row in df.iterrows():
        string = (str(row['Short Description'])+(row['Summary']))
        label = str(row['Classification'])
        date = str(row['Incident Date'])
        counter = str(row['Incident Number'])
        if label != 'nan':
            classified.date.append(date)
            classified.data.append(string)
            classified.label.append(label)
            classified.id.append(counter)
        unseen.date.append(date)        
        unseen.data.append(string)
        unseen.label.append(label)
        unseen.id.append(counter)

    # train:test ratio
    train.data, test.data, train.label, test.label = train_test_split(classified.data, 
                                                                      classified.label, 
                                                                      test_size=0.2, 
                                                                      random_state=5) 
    return train, test, unseen

def create_clf(clf_type):
    if clf_type == "NB":
        text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB())])
        parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                      'tfidf__use_idf': (True, False),
                      'clf__alpha': (1e-2, 1e-3)}
    elif clf_type == "SVM":
        text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                             ('tfidf', TfidfTransformer()),
                             ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                                   alpha=1e-3,  
                                                   max_iter=5000,
                                                   tol=1e-3,
                                                   random_state=42))
                             ])
        parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                      'tfidf__use_idf': (True, False),
                      'clf__alpha': (1e-2, 1e-3)}    

    return text_clf, parameters

def model(MODEL):
    train, test, unseen = read_csv(path, scramble=True) 
        
    text_clf, parameters = create_clf(MODEL);
    text_clf = text_clf.fit(train.data, train.label)

    predicted = text_clf.predict(test.data)
    print("Non Grid Test acc:",  np.mean(predicted == test.label))

    #gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1, cv=3, iid=False)
    #gs_clf = gs_clf.fit(train.data, train.label)
    #predicted = gs_clf.predict(test.data)
    #print("Grid Test acc:",  np.mean(predicted == test.label))

    # Testing against unseen 
    prediction = text_clf.predict(unseen.data)
    
    #print("Grid Test acc:",  np.mean(predicted == test.label))
  
    df = pd.DataFrame({'number' : unseen.id ,'desc' : unseen.data, 'category' : prediction, 'date' : unseen.date}) 
    df.to_csv('designclass_modified.csv')

if __name__ == "__main__":
    path = "incidents.csv"    
    MODEL = "NB"
    model(MODEL)
    excel_col = ["desc","design","date","number"]
