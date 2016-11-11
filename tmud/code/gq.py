__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import datetime
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import random
import zipfile
import time
import shutil
from sklearn.metrics import log_loss
from categorize import get_categ
import json
import time
from pylab import show

random.seed(2016)

def run_xgb(train, test, features, target, random_state=0):
    eta = 0.02
    max_depth = 5 
    subsample = 0.75
    colsample_bytree = 0.7
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "multi:softprob",
        "num_class": 12,
        "booster" : "gbtree",
        "eval_metric": "mlogloss",
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state,
    }
    num_boost_round = 500*2
    early_stopping_rounds = 50
    test_size = 0.3

    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    print('Length train:', len(X_train.index))
    print('Length valid:', len(X_valid.index))
    y_train = X_train[target]
    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

    print "importance of feathure"
    xgb.plot_importance(gbm)
    show()


    #time.sleep(60*5)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration)
    score = log_loss(y_valid.tolist(), check)

    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_iteration)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction.tolist(), score


def create_submission(score, test, prediction):
    # Make Submission
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('device_id,F23-,F24-26,F27-28,F29-32,F33-42,F43+,M22-,M23-26,M27-28,M29-31,M32-38,M39+\n')
    total = 0
    test_val = test['device_id'].values
    for i in range(len(test_val)):
        str1 = str(test_val[i])
        for j in range(12):
            str1 += ',' + str(prediction[i][j])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()


def map_column(table, f):
    labels = sorted(table[f].unique())
    mappings = dict()
    for i in range(len(labels)):
        mappings[labels[i]] = i
    table = table.replace({f: mappings})
    return table


def read_train_test():
    # Events
    # print('Read events...')
    # events = pd.read_csv("../input/events.csv", dtype={'device_id': np.str})
    # events['counts'] = events.groupby(['device_id'])['event_id'].transform('count')
    # #events_small = events[['device_id', 'counts']].drop_duplicates('device_id', keep='first')
    #
    # # App events
    # app_events = pd.read_csv("../input/app_events.csv",dtype={"device_id": np.str})
    # del app_events['is_installed']
    # del app_events['is_active']
    #
    # # App labels
    # # print 'Read app label...'
    # # app_label = pd.read_csv("../input/app_labels.csv", dtype={'device_id': np.str})
    # # app_events = pd.merge(app_events, app_label, how='left', on='app_id', left_index=True)
    # # app_events.fillna(0, inplace=True)
    # # app_events['game'] = [1 if (item<94 and item>1) else 0 for item in app_events['label_id']]
    # # app_events['game'] = app_events['game'] * app_events['is_active']
    # app_label = get_categ()
    # app_events = pd.merge(app_events, app_label, how='left', on='app_id')
    #
    # # Events_small
    # #app_events = app_events[['event_id', 'game']]
    # app_events['game_count'] = app_events.groupby(['event_id'])['Games'].transform('sum')
    # app_events['video_count'] = app_events.groupby(['event_id'])['Video'].transform('sum')
    # app_events['edu_count'] = app_events.groupby(['event_id'])['Education'].transform('sum')
    # app_events['sport_count'] = app_events.groupby(['event_id'])['Sports'].transform('sum')
    # app_events['music_count'] = app_events.groupby(['event_id'])['Music'].transform('sum')
    # app_events = app_events[['event_id', 'game_count', 'video_count', 'edu_count', 'sport_count', 'music_count']].drop_duplicates('event_id', keep='first')
    #
    # events = pd.merge(events, app_events, how='left', on='event_id', left_index=True)
    # #events['game_count'] = events.groupby(['device_id'])['game'].transform('sum')
    # events = events.drop(['game'], axis=1)
    # events_small = events[['device_id', 'counts', 'game_count', 'video_count', 'edu_count', 'sport_count', 'music_count']].drop_duplicates('device_id', keep='first')
    #
    # Phone brand
    print('Read brands...')
    pbd = pd.read_csv("../input/phone_brand_device_model.csv", dtype={'device_id': np.str})
    pbd.drop_duplicates('device_id', keep='first', inplace=True)
    pbd = map_column(pbd, 'phone_brand')
    pbd = map_column(pbd, 'device_model')

    # Read tmud mainjson
    l = []
    with open("../input/tmud_main_080602.json") as ff:
        for i in ff.readlines():
            l.append(json.loads(i))

    events_small = pd.DataFrame(l)
    del events_small['device_id_33']
    print events_small.head()

    # Train
    print('Read train...')
    train = pd.read_csv("../input/gender_age_train.csv", dtype={'device_id': np.str})
    train = map_column(train, 'group')
    train = train.drop(['age'], axis=1)
    train = train.drop(['gender'], axis=1)
    train = pd.merge(train, pbd, how='left', on='device_id', left_index=True)
    train = pd.merge(train, events_small, how='left', on='device_id', left_index=True)
    train.fillna(-1, inplace=True)

    # Test
    print('Read test...')
    test = pd.read_csv("../input/gender_age_test.csv", dtype={'device_id': np.str})
    test = pd.merge(test, pbd, how='left', on='device_id', left_index=True)
    test = pd.merge(test, events_small, how='left', on='device_id', left_index=True)
    test.fillna(-1, inplace=True)

    # Features
    features = list(test.columns.values)
    features.remove('device_id')
    # features.remove('Sports')
    # features.remove('Sports_active_25_time')
    # features.remove('Sports_active_50_time')
    # features.remove('Sports_active_75_time')
    # features.remove('Music')



    print "features",features

    
    return train, test, features


train, test, features = read_train_test()
print('Length of train: ', len(train))
print('Length of test: ', len(test))
print('Features [{}]: {}'.format(len(features), sorted(features)))
test_prediction, score = run_xgb(train, test, features, 'group')
print("LS: {}".format(round(score, 5)))
create_submission(score, test, test_prediction)

