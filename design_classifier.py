# Graphing
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

# Helpers
import numpy as np
import pandas as pd
from joblib import dump, load
from collections import OrderedDict

# Machine Learning
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss

class Data:
    def __init__(self):
        self.data = []
        self.label = []
        self.date = []
        self.id = []
            
def read_csv(file_path):
    #df = pd.read_csv(file_path).sample(frac=1)
    df = pd.read_csv(file_path)
    train = Data()
    test = Data()
    unseen = Data()

    for index, row in df.iterrows():
        string = (str(row['Brief Description of Incident']) + " " + 
                  str(row['Incident Intial Summary']))
        label = str(row['Safety In Design (operational/maintenance functions)'])
        date = str(row['Incident Date'])
        counter = str(row['Counter'])
        if label == 'y' or label == 'n':
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
        text_clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                    subsample=0.8, nthread=10, learning_rate=0.1)
        parameters = {}
        #clf.fit(xtrain_tfv.tocsc(), ytrain)
        #predictions = clf.predict_proba(xvalid_tfv.tocsc())
        
    return text_clf, parameters

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

    no_dict = OrderedDict(sorted(no_dict.items()))
    yes_dict = OrderedDict(sorted(yes_dict.items()))

    non_design = go.Bar(
        x=[key for key in no_dict],
        y=[no_dict[key] for key in no_dict],
        name='Non-Design Related',
	marker=dict(
	    color='rgb(55, 83, 109)'
	)
    )
    design = go.Bar(
        x=[key for key in yes_dict],
        y=[yes_dict[key] for key in yes_dict],
        name='Design Related',
	marker=dict(
	    color='rgb(26, 118, 255)'
	)
    )
    
    data = [non_design, design]
    layout = go.Layout(
        title='Sydney Water Design vs Non-Design Related Incidents (2013-Present)',
        barmode='group',
	xaxis = {'title': 'Years'},
	yaxis = {'title': 'Number of Incidents'}
    )

    fig = go.Figure(data=data, layout=layout)
    """ To save to file, py.iplot(fig, filename='whatever_name') """
    # For Jupyter uncomment line below
    #py.iplot(fig)
    # For Python uncomment line below
    py.plot(fig)

def find_best_clf(text_clf):
    best_acc = 0
    for _ in range(1000):
        text_clf = text_clf.fit(train.data, train.label)
        predicted = text_clf.predict(test.data)
        acc = np.mean(predicted == test.label)
        print("Test acc:",  np.mean(predicted == test.label))
        if acc > best_acc:
            best_acc = acc
            best_clf = text_clf
    print("Best accuracy:", best_acc)
    dump(best_clf, 'best_design_clf.joblib')


if __name__ == "__main__":
    path = "SWIRL_Incident_Details_design.csv"
    train, test, unseen = read_csv(path)   
    MODEL = "SVM"

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
    render_graph(unseen)
