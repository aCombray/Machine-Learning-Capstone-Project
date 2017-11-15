
# coding: utf-8

# In[1]:

# Python library
import os
import sys
import operator
import random
import math
import re
import json

# database
import pandas as pd
import numpy as np
import glob


# data wrangler
from scipy import sparse
from sklearn import model_selection, preprocessing, ensemble
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud

# pipeline
from sklearn.pipeline import Pipeline

# model selection
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, KFold, GridSearchCV


# parameter tuning
from bayes_opt import BayesianOptimization



# classification
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neighbors import KNeighborsRegressor as KNR

# metrics
from sklearn.metrics import classification_report, accuracy_score, log_loss



class BO(object):
    class Floor(object):
        '''
        Given a list of integer parameters int_params, change the values of the keys in int_params to integers.
        '''
        def __init__(self, int_params):
            self._int_params = int_params
            self.m = len(int_params)
            
        def convert(self, params):
            for i in range(self.m):
                params[self._int_params[i]] = int(params[self._int_params[i]])
            return params
        
    '''
    int_params is the list of the integer params names
    float_params in the list of float params names
    params_interval is the dictionary {key: (a, b)}
    Training data must only involve int or float
    '''
    def __init__(self, clf_name, metric, X_train, y_train, chosen_params={}, K=5, is_random=False, seed=0, pipelines=None):
        self._clf = clf_name
        self.K=K
        self.X=X_train
        self.y=y_train
        self.metric = metric
        self.trained_clf=None
        self._floor=None
        self.tuned_params=chosen_params
        self.seed=seed
        self.pipeline=pipelines
        if is_random:
            self.tuned_params['random_state']=seed
    
    def cv(self, **params):
        params_dict = dict(self.tuned_params)
        self._floor.convert(params)
        params_dict.update(params)
        scores=[]
        pipe_params={}
        for key, value in params.iteritems():
            k = 'clf__'+key
            pipe_params[k]=value
        kf = KFold(n_splits=self.K, shuffle=True, random_state=self.seed)
        for train_index, test_index in kf.split(self.X):
            X_train, X_test=self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test=self.y.iloc[train_index], self.y.iloc[test_index]
            self.pipeline.set_params(**pipe_params)
            self.pipeline.fit(X_train, y_train)
            y_pred = self.pipeline.predict_proba(X_test)
            scores.append(log_loss(y_test, y_pred, labels=[0,1,2])) 
        val = -np.mean(scores)
        return val

    def find_params(self, int_params, params_intervals, n_iter=10,init_points=20, explore=None, pipeline=None):
        self._floor = self.Floor(int_params)
        clf=self._clf(**self.tuned_params)
        if pipeline:
            self.pipeline=pipeline
        if self.pipeline is None:
            self.pipeline=Pipeline([('clf', self._clf(**self.tuned_params))])
        print self.pipeline.named_steps
        clfBO = BayesianOptimization(
            self.cv,
            params_intervals
        )
        gp_params = {"alpha": 1e-5}
        if explore is not None:
            clfBO.explore(explore)
        clfBO.maximize(n_iter=n_iter,init_points=init_points, **gp_params)
        print('-' * 53)
        print('Final Results')
        print('{}: {}'.format(self._clf, clfBO.res['max']['max_params']))
        params=clfBO.res['max']['max_params']
        self._floor.convert(params)
        self.tuned_params.update(params)
        self.trained_clf = self._clf(**self.tuned_params)




