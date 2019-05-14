#!/usr/bin/env python
# coding: utf-8

# # Importing modules

# In[27]:


from IPython.display import display, HTML

# Visualisation tools
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns

# Processing tools
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd

# sklearn models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# sklearn tools
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# imblearn samplers
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN


# In[67]:


def read_csv(file_path, scramble=False):
    if scramble:
        return (pd.read_csv(file_path).sample(frac=1));
    return pd.read_csv(file_path)

def pre_process(df, train=True):
    if train:
        df = df.dropna(subset=['Classification'])
        
    df = df.fillna('')
    df['Incident Summary'] = df['Short Description'] + " " +                              df['Summary'] + " " +                              df['Root Cause'] + " " +                              df['Mechanism of Injury Description']
    if train:
        #show_class_distribution(df)
        df = df[['Incident Date', 'Incident Summary', 'Classification']]
        df['category_id'] = df['Classification'].factorize()[0]
        category_id_df = df[['Classification', 'category_id']].drop_duplicates().sort_values('category_id')
        category_to_id = dict(category_id_df.values)
        id_to_category = dict(category_id_df[['category_id', 'Classification']].values)
    else:
        df = df[['Incident Number', 'Incident Date', 'Incident Summary']]
    
    if train: 
        return (df, category_id_df, category_to_id, id_to_category)
    else:
        return (df)

def show_class_distribution(df, class_name):
    fig = plt.figure()
    df.groupby(class_name)[class_name].count().sort_values(ascending=False).plot.bar()
    fig.tight_layout()
    #fig.savefig('images/cat_clf.png')
    plt.show()

def train_svm(df, category_id_df, category_to_id, id_to_category, analysis=False):
    tfidf = TfidfVectorizer(sublinear_tf=False, min_df=3,
                            ngram_range=(1,3), stop_words='english')
    features = tfidf.fit_transform(df['Incident Summary']).toarray()
    labels = df.category_id
    
    # Show features from tfidf
    #show_features(tfidf, features, labels, category_to_id)

    # Test different classifier models
    #test_models(df, features, labels)
    
    model = LinearSVC(C=1,random_state=1)
    X_train, X_test, y_train, y_test, indices_train, indices_test =                 train_test_split(features, labels, df.index, test_size=0.3, random_state=0)
    
    # Undersample 'Other' class and oversample others
    sm = RandomOverSampler(random_state=1)
    X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

    # Fit model with resampled data
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)

    
    accuracy = accuracy_score(y_pred, y_test)
    print("Model Accuracy:", accuracy)

    if analysis:
        # Show model metrics
        show_metrics(y_test, y_pred, df)

        # Confusion matrix
        conf_mat = confusion_matrix(y_test, y_pred)
        sns_plot = sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=category_id_df['Classification'].values, 
                yticklabels=category_id_df['Classification'].values)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')

        #fig = sns_plot.get_figure()
        #fig.tight_layout()
        #fig.savefig('images/conf_matrix.png')
        #plt.show()

        # Display table where number of incorrectly classified cases > 2
        '''
        pd.set_option('display.max_colwidth', -1)
        for predicted in category_id_df.category_id:
            for actual in category_id_df.category_id:
                if predicted != actual and conf_mat[actual, predicted] >= 2:
                  print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))
                  display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['Classification', 'Incident Summary']])
                  print('')

        # Show models features
        model.fit(features, labels)
        N = 2
        for Product, category_id in sorted(category_to_id.items()):
          indices = np.argsort(model.coef_[category_id])
          feature_names = np.array(tfidf.get_feature_names())[indices]
          unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
          bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
          print("# '{}':".format(Product))
          print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
          print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))
        '''
    
    return(tfidf, model)

def predict(df, tfidf, model, id_to_category, output_csv=False):
    df = pre_process(df, train=False)
    features = tfidf.transform(df['Incident Summary']).toarray()

    y_pred = model.predict(features)
    df['predicted'] = [id_to_category[pred] for pred in y_pred]

    show_class_distribution(df, 'predicted')
    
    if output_csv:
        df.to_csv('yay.csv')


def show_features(tfidf, features, labels, category_to_id):
    N = 2
    for Classification, category_id in sorted(category_to_id.items()):
        features_chi2 = chi2(features, labels == category_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        print("# '{}':".format(Classification))
        print("Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
        print("Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
        
def test_models(df, features, labels):
    models = [
        RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
        DecisionTreeClassifier(),
        KNeighborsClassifier(),
        LinearSVC(),
        SGDClassifier(loss='hinge', class_weight='balanced'),
        MultinomialNB(),
        MLPClassifier(),
        #GradientBoostingClassifier(),
        LogisticRegression(random_state=0),
    ]
    CV = 20
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
      model_name = model.__class__.__name__
      accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
      for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

    
    sns.boxplot(x='model_name', y='accuracy', data=cv_df)
    sns_plot = sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
                  size=8, jitter=True, edgecolor="gray", linewidth=2)
    #sns_plot = sns_plot.get_figure()
    #sns_plot.savefig('images/clf_performance.png')
    plt.show()
    print(cv_df.groupby('model_name').accuracy.mean())

def show_metrics(y_test, y_pred, df):
    print(classification_report(y_test, y_pred, target_names=df['Classification'].unique()))

if __name__ == "__main__":
    path = "incidents.csv"    
    df = read_csv(path)
    labelled_df, category_id_df, category_to_id, id_to_category = pre_process(df)
    tfidf, model = train_svm(labelled_df, category_id_df, 
                             category_to_id, id_to_category, analysis=False)
    predict(df, tfidf, model, id_to_category, output_csv=False)
    
