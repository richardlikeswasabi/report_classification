import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

from collections import OrderedDict
import numpy as np
import pandas as pd

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
        self.id = []
            
def read_csv(file_path):
    df = pd.read_csv(file_path).sample(frac=1)
    train = Data()
    test = Data()
    all_data = Data()

    for index, row in df.iterrows():
        string = (str(row['Brief Description of Incident']) + " " + 
                  str(row['Incident Intial Summary']) + " " +
                  str(row['Nature of Injury Description']) + " " +
                  str(row['Root Cause']))
        label = str(row['Safety In Design (operational/maintenance functions)'])
        date = str(row['Incident Date'])
        if label == 'y' or label == 'n':
            train.date.append(date)
            train.data.append(string)
            train.label.append(label)
        all_data.date.append(date)        
        all_data.data.append(string)
        all_data.label.append(label)

    # train:test ratio
    split_ratio = 0.8  
    data_split = int(len(train.data) * split_ratio)

    test.date = train.date[data_split+1:]
    test.data = train.data[data_split+1:]
    test.label = train.label[data_split+1:]

    train.date = train.date[:data_split]
    train.data = train.data[:data_split]
    train.label = train.label[:data_split]   

    return train, test, all_data

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

def output_excel(all_data, file_name):
    excel = []        

    for i in range(len(all_data.data)):
        excel.append(all_data.data[i])
        excel.append(all_data.predicted[i])
        excel.append(all_data.date[i])
            
    excel_col = ["desc","design","date"]
    num = np.array(excel)
    reshaped = num.reshape(int(len(excel)/3),3)
    df1 = pd.DataFrame(reshaped, columns=excel_col)
    df1.to_csv(file_name)

if __name__ == "__main__":
    path = "design_incidents.csv"
    train, test, all_data = read_csv(path)   
    MODEL = "SVM"

    text_clf, parameters = create_clf(MODEL);
    text_clf = text_clf.fit(train.data, train.label)

    predicted = text_clf.predict(test.data)
    print("Accuracy:",  np.mean(predicted == test.label))

    # Using grid search
    #gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1, cv=3, iid=False)
    #gs_clf = gs_clf.fit(train.data, train.label)
    #predicted = gs_clf.predict(test.data)
    #print("Grid Test acc:",  np.mean(predicted == test.label))

    all_data.predicted = text_clf.predict(all_data.data)
    # Uncomment to output excel file
    #output_excel(all_data, "design_classified.csv")

    # Uncomment to render graph
    #render_graph(all_data)

