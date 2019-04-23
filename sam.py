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



from sklearn.model_selection import GridSearchCV

from sklearn.metrics import log_loss

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import re
import nltk
def Tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    porter_stemmer=nltk.PorterStemmer()
    words = [porter_stemmer.stem(word) for word in words]
    return words


from sklearn.base import BaseEstimator, TransformerMixin
class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, (train.data), y=None):
        return self
    def transform(self, (train.data):
        return (train.data)[self.field]
    '''
class NumberSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[[self.field]]
    '''
    
class Data:

    def __init__(self):

        self.data = []

        self.label = []

        self.date = []

        self.id = []

            

def read_csv(file_path):

    df = pd.read_csv(file_path).sample(frac=1)

    train = Data()

    test = Data()

    unseen = Data()



    for index, row in df.iterrows():

        string = (str(row['Short Description']) + " " + 
                str(row['Mechanism of Injury Description']) + " " +    
                str(row['Investigation Root Cause Description']) + " " +    
                str(row['Summary']))

        label = str(row['Classification'])

        date = str(row['Incident Date'])

        counter = str(row['Incident Number'])

        if label != "nan" :

            train.date.append(date)

            train.data.append(string)

            train.label.append(label)

            train.id.append(counter)

        unseen.date.append(date)        

        unseen.data.append(string)

        unseen.label.append(label)

        unseen.id.append(counter)



    # train:test ratio

    split_ratio = 0.8  

    data_split = int(len(train.data) * split_ratio)



    test.date = train.date[data_split+1:]

    test.data = train.data[data_split+1:]

    test.label = train.label[data_split+1:]



    train.date = train.date[:data_split]

    train.data = train.data[:data_split]

    train.label = train.label[:data_split]   



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

    elif clf_type == "XGB":

        text_clf = Pipeline([
            ('features', FeatureUnion([
                ('text', Pipeline([
                    ('colext', TextSelector('Text')),
                    ('tfidf', TfidfVectorizer(tokenizer=Tokenizer, stop_words='english',
                         min_df=.0025, max_df=0.25, ngram_range=(1,3))),
            ('svd', TruncatedSVD(algorithm='randomized', n_components=300)), #for XGB
            ])),
            ('words', Pipeline([
                ('wordext', NumberSelector('TotalWords')),
                ('wscaler', StandardScaler()),
            ])),
        ])),
        ('clf', XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1)),
        #    ('clf', RandomForestClassifier()),
        ])
        parameters = {}

    return text_clf, parameters



if __name__ == "__main__":

    path = "incidents.csv"

    train, test, unseen = read_csv(path)   

    MODEL = "XGB"

    text_clf, parameters = create_clf(MODEL);

    text_clf = text_clf.fit(train.data, train.label)
    
    predicted  = text_clf.predict(train.data)

    print("Training acc:",  np.mean(predicted == train.label))



    predicted = text_clf.predict(test.data)

    print("Non Grid Test acc:",  np.mean(predicted == test.label))

   

    # Using grid search

    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1, cv=3, iid=False)

    gs_clf = gs_clf.fit(train.data, train.label)

    predicted = gs_clf.predict(test.data)

    print("Grid Test acc:",  np.mean(predicted == test.label))



    # Testing against unseen 

    unseen.predicted = text_clf.predict(unseen.data)

    '''

    #testing the train data

    train.predicted = text_clf.predict(train.data)

    print("trained", np.mean(train.predicted == train.label))

    '''    

    excel = []        

    no, yes, x = 0, 0, 0



    for i in range(len(test.data)):

        if predicted[i] != test.label[i]:

            #print(test.data[i], '|', predicted[i], '|', test.label[i])

            predicted[i] = test.label[i]

            x += 1    



    for i in range(len(unseen.data)):

        excel.append(unseen.data[i])

        excel.append(unseen.predicted[i])

        excel.append(unseen.date[i])

        excel.append(unseen.id[i])

            

    excel_col = ["desc","design","date","number"]

    num = np.array(excel)

    reshaped = num.reshape(int(len(excel)/4),4)

    df1 = pd.DataFrame(reshaped, columns=excel_col)

    df1.to_csv('designclass_modified.csv')

   
