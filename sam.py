import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

class Data:
    def __init__(self):
        self.data = []
        self.label = []
        self.date = []

def read_csv(file_path):
    df = pd.read_csv(file_path)
    print(df.columns)
    #df = df[['Short Description', 'Incident Type', 'Root Cause', 'Design Related Potential']]
    train = Data()
    test = Data()
    unseen = Data()
    
    # data = all data
    # train.data = subset of data wih tags
    # train.label = corress
    for index, row in df.iterrows():
        string = (str(row['Short Description']) + " " + 
                  str(row['Incident Type']) + " " + 
                  str(row['Root Cause']))
        label = str(row['Design Related Potential'])
        date = str(row['Incident Date'])

        if label == 'y' or label == 'n':
            train.date.append(date)
            train.data.append(string)
            train.label.append(label)

        unseen.date.append(date)
        unseen.data.append(string)
        unseen.label.append(label)

    # split train + test data
    TRAIN_TEST = 60
    test.date = train.date[TRAIN_TEST+1:]
    test.data = train.data[TRAIN_TEST+1:]
    test.label = train.label[TRAIN_TEST+1:]

    train.date = train.date[:TRAIN_TEST]
    train.data = train.data[:TRAIN_TEST]
    train.label = train.label[:TRAIN_TEST]

    return train, test, unseen

def render_graph(data):
    yes_dict = {}
    no_dict = {}
    for i in range(len(data.data)):
        if data.label[i] == 'y':
            if len(data.date[i].split('/')) != 3:
                continue

            date = data.date[i].split('/')[2]

            if date not in yes_dict:
                yes_dict[date] = 1
            else:
                yes_dict[date] += 1
        elif data.label[i] == 'n':
            if len(data.date[i].split('/')) != 3:
                continue

            date = data.date[i].split('/')[2]

            if date not in no_dict:
                no_dict[date] = 1
            else:
                no_dict[date] += 1


   
    plt.xlabel('Date')
    plt.ylabel('Incidents')
    plt.bar([key for key in no_dict], [no_dict[key] for key in no_dict], label="Non-design related")
    plt.bar([key for key in yes_dict], [yes_dict[key] for key in yes_dict], label="Design related")
    plt.legend(loc='upper left')
    plt.title("Design vs Non-design Related Incidents")
    plt.show()





if __name__ == "__main__":
    path = "incidents.csv"
    train, test, unseen = read_csv(path)
    
    MODEL = "NB"
    if MODEL == "NB":
        text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB())])
        parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                      'tfidf__use_idf': (True, False),
                      'clf__alpha': (1e-2, 1e-3)}
    elif MODEL == "SVM":
        text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                             ('tfidf', TfidfTransformer()),
                             ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                                   alpha=1e-3,  
                                                   random_state=42))
                             ])
        parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                      'tfidf__use_idf': (True, False),
                      'clf__alpha': (1e-2, 1e-3)}

    # Using purely BN/SVM
    text_clf = text_clf.fit(train.data, train.label)
    predicted = text_clf.predict(test.data)
    print("non grid search", np.mean(predicted == test.label))
    
    # Using grid search
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1, cv=3, iid=False)
    gs_clf = gs_clf.fit(train.data, train.label)
    predicted = gs_clf.predict(test.data)
    print("grid search", np.mean(predicted == test.label))


    #for i in range(len(test.data)):
    #    print(test.data[i], '|', predicted[i], '|', test.label[i], '|', test.date[i])

    # Testing against unseen 
    unseen.label = text_clf.predict(unseen.data)
    #print(np.mean(predicted == test.label))
    no, yes = 0, 0
    #for i in range(len(unseen.data)):
    #    print(unseen.data[i], '|', unseen_predicted[i])
    #    if unseen_predicted[i] == 'y':
    #        yes += 1
    #    else:
    #        no += 1
    #render_graph([unseen[i] for i in range(len(unseen_predicted)) if unseen_predicted[i] == 'y'])
    render_graph(unseen)

    print("Yes:", yes, "No:" , no)
    
%matplotlib notebook

import matplotlib.pyplot as plt

def render_graph(data):

    yes_dict = {}
    no_dict = {}
    for i in range(len(data.data)):
        if data.label[i] == 'y':
            if len(data.date[i].split('/')) != 3:
                continue

            date = data.date[i].split('/')[2]

            if date not in yes_dict:
                yes_dict[date] = 1
            else:
                yes_dict[date] += 1

        elif data.label[i] == 'n':
            if len(data.date[i].split('/')) != 3:
                continue

            date = data.date[i].split('/')[2]

            if date not in no_dict:

                no_dict[date] = 1
            else:
                no_dict[date] += 1

  

    plt.xlabel('Date')
    plt.ylabel('Incidents')
    plt.bar([key for key in no_dict], [no_dict[key] for key in no_dict], label="Non-design related")
    plt.bar([key for key in yes_dict], [yes_dict[key] for key in yes_dict], label="Design related")
    plt.legend(loc='upper left')
    plt.title("Design vs Non-design Related Incidents")
    plt.show()
