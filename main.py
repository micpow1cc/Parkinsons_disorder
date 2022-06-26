import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


def load_data():
    global features, labels
    df = pd.read_csv("C:\\Users\\micpo\\Downloads\\parkinsons.data")
    pd.set_option('expand_frame_repr', False)
    features = df.loc[:, df.columns != 'status'].values[:, 1:]
    labels = df.loc[:, 'status'].values
    print(labels[labels == 1].shape[0], labels[labels == 0].shape[0])


def Model():
    global cv
    scaler = MinMaxScaler((-1, 1))
    x = scaler.fit_transform(features)
    y = labels
    # split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
    parameters = {
        "n_estimators": [5, 10, 50, 100, 250],
        "max_depth": [2, 4, 8, 16, 32, None]

    }
    model = XGBClassifier(max_depth = 2,n_estimators=250)
    model.fit(x_train, y_train)
    #cv = GridSearchCV(model, parameters, cv=5,scoring = 'roc_auc')
    #cv.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(y_pred, "-",y_test)
    tn,fp,fn,tp = confusion_matrix(y_test, y_pred).ravel()
    print(tn,fp,fn,tp)
    print((tp+tn)/(tn+fp+fn+tp))
    # 97,4% roc auc scoring




def display(results): # metoda pozwalajaca wyswietlic najlepsze parametry max_depth i n_estimators
    print(f'Best parameters are: {results.best_params_}')
    print("\n")
    mean_score = results.cv_results_['mean_test_score']
    std_score = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean,std,params in zip(mean_score,std_score,params):
        print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')


load_data()
Model()
#display(cv)
