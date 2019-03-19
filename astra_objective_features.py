# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:38:40 2019

@author: tanma
"""

import pandas as pd
import numpy as np

data = pd.read_csv("cumulative.csv")

x = data.drop(['rowid','kepid','kepoi_name','kepler_name','koi_pdisposition','koi_disposition','koi_tce_delivname','koi_teq_err1','koi_teq_err2'], axis = 1)
y = data['koi_disposition']
y_copy = list(y)

for i in range(len(y)):
    if y_copy[i] == "CONFIRMED":
        y_copy[i] = 0
    else:
        y_copy[i] = 1

y = pd.DataFrame({'Label': y_copy}, index = x.index)

features = x.columns

param = {
       'bagging_freq': 5,
       'bagging_fraction': 0.32,
        'boost_from_average':'false',
        'boost': 'gbdt',
        'feature_fraction': 0.045,
        'learning_rate': 0.01,
        'max_depth': -1,  
        'metric':'binary_logloss',
        'min_data_in_leaf': 80,
        'min_sum_hessian_in_leaf': 10.0,
        'num_leaves': 13,
        'num_threads': 8,
        'tree_learner': 'serial',
        'lambda_l2':0.1,
        'objective': 'binary', 
        'verbosity': 1
    }

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score,roc_auc_score
import os

num_round = 20000
folds = KFold(n_splits=10, shuffle=False, random_state=32)
oof = np.zeros(len(data))

for fold_no, (trn_idx, val_idx) in enumerate(folds.split(x.values, y.values)):
    print("Fold {}".format(fold_no))
    trn_data = lgb.Dataset(x.iloc[trn_idx], label=y.iloc[trn_idx])
    val_data = lgb.Dataset(x.iloc[val_idx], label=y.iloc[val_idx])
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=500,early_stopping_rounds = 5000)
    oof[val_idx] = clf.predict(x.iloc[val_idx], num_iteration=clf.best_iteration)
print("CV score: {:<8.5f}".format(roc_auc_score(y, oof)))