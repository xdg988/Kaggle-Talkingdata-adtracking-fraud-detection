"""
A non-blending lightGBM model that incorporates portions and ideas from various public kernels.
"""

import pandas as pd
import time
import numpy as np
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
import os

def do_count( df, group_cols, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Aggregating by ", group_cols , '...' )
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

def do_countuniq( df, group_cols, counted, agg_name, agg_type='uint16', show_max=False, show_agg=True ):
    if show_agg:
        print( "Counting unqiue ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

def add_next_click(df):
    print('Extracting next click1...')
    D = 2**26
    df['category'] = (df['ip'].astype(str) + "_" + df['app'].astype(str) + "_" + df['device'].astype(str) \
                      + "_" + df['os'].astype(str)).apply(hash) % D
    click_buffer = np.full(D, 3000000000, dtype=np.uint32)
    df['epochtime'] = df['click_time'].astype(np.int64) // 10 ** 9
    next_clicks = []
    for category, time in zip(reversed(df['category'].values), reversed(df['epochtime'].values)):
        next_clicks.append(click_buffer[category] - time)
        click_buffer[category] = time
    del click_buffer
    df['next_click'] = list(reversed(next_clicks))
    df['next_click'] = df['next_click'].astype('uint32')
    df.drop(['category', 'epochtime'], axis=1, inplace=True)
    return (df)
    
def add_next_click2(df):
    print('Extracting next click2...')
    D = 2**26
    df['category'] = (df['ip'].astype(str) + "_" + df['app'].astype(str) + "_" + df['device'].astype(str) \
                      + "_").apply(hash) % D
    click_buffer = np.full(D, 3000000000, dtype=np.uint32)
    df['epochtime'] = df['click_time'].astype(np.int64) // 10 ** 9
    next_clicks = []
    for category, time in zip(reversed(df['category'].values), reversed(df['epochtime'].values)):
        next_clicks.append(click_buffer[category] - time)
        click_buffer[category] = time
    del click_buffer
    df['next_click2'] = list(reversed(next_clicks))
    df['next_click2'] = df['next_click2'].astype('uint32')
    df.drop(['category', 'epochtime'], axis=1, inplace=True)
    del next_clicks
    return (df)

def add_next_click3(df):
    print('Extracting next click3...')
    D = 2**26
    df['category'] = (df['ip'].astype(str) + "_" + df['app'].astype(str) + "_" + df['os'].astype(str) \
                      + "_").apply(hash) % D
    click_buffer = np.full(D, 3000000000, dtype=np.uint32)
    df['epochtime'] = df['click_time'].astype(np.int64) // 10 ** 9
    next_clicks = []
    for category, time in zip(reversed(df['category'].values), reversed(df['epochtime'].values)):
        next_clicks.append(click_buffer[category] - time)
        click_buffer[category] = time
    del click_buffer
    df['next_click3'] = list(reversed(next_clicks))
    df['next_click3'] = df['next_click3'].astype('uint32')
    df.drop(['category', 'epochtime'], axis=1, inplace=True)
    del next_clicks
    return (df)
    
def add_next_click4(df):
    print('Extracting next click4...')
    D = 2**26
    df['category'] = (df['ip'].astype(str) + "_" + df['app'].astype(str) + "_" + df['channel'].astype(str) \
                      + "_" + df['os'].astype(str)).apply(hash) % D
    click_buffer = np.full(D, 3000000000, dtype=np.uint32)
    df['epochtime'] = df['click_time'].astype(np.int64) // 10 ** 9
    next_clicks = []
    for category, time in zip(reversed(df['category'].values), reversed(df['epochtime'].values)):
        next_clicks.append(click_buffer[category] - time)
        click_buffer[category] = time
    del click_buffer
    df['next_click4'] = list(reversed(next_clicks))
    df['next_click4'] = df['next_click4'].astype('uint32')
    df.drop(['category', 'epochtime'], axis=1, inplace=True)
    del next_clicks
    return (df)
    
def add_next_click5(df):
    print('Extracting next click5...')
    D = 2**26
    df['category'] = (df['ip'].astype(str) + "_" + df['app'].astype(str) + "_" + df['channel'].astype(str) \
                      + "_" + df['os'].astype(str)+"_" + df['device'].astype(str)).apply(hash) % D
    click_buffer = np.full(D, 3000000000, dtype=np.uint32)
    df['epochtime'] = df['click_time'].astype(np.int64) // 10 ** 9
    next_clicks = []
    for category, time in zip(reversed(df['category'].values), reversed(df['epochtime'].values)):
        next_clicks.append(click_buffer[category] - time)
        click_buffer[category] = time
    del click_buffer
    df['next_click5'] = list(reversed(next_clicks))
    df['next_click5'] = df['next_click5'].astype('uint32')
    df.drop(['category', 'epochtime'], axis=1, inplace=True)
    del next_clicks
    return (df)

predictors=[]
debug= 0
if debug:
    print('*** debug parameter set: this is a test run for debugging purposes ***')

def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                 feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.2,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4,
        'verbose': 0,
        'metric':metrics
    }

    lgb_params.update(params)

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}

    bst1 = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgvalid], 
                     valid_names=['valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=feval)

    print("\nModel Report")
    print("bst1.best_iteration: ", bst1.best_iteration)
    print(metrics+":", evals_results['valid'][metrics][bst1.best_iteration-1])

    return (bst1,bst1.best_iteration)

def DO(frm,to,fileno):
    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32',
            }

    print('loading train data...',frm,to)
    train_df = pd.read_csv("D:/machine-learning/kaggle/talkingdata/train.csv", parse_dates=['click_time'], skiprows=range(1,frm), nrows=to-frm, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])

    print('loading test data...')
    if debug:
        test_df = pd.read_csv("D:/machine-learning/kaggle/talkingdata/test.csv", nrows=100000, parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    else:
        test_df = pd.read_csv("D:/machine-learning/kaggle/talkingdata/test.csv", parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

    len_train = len(train_df)
    train_df=train_df.append(test_df)
    
    del test_df
    gc.collect()
    
    print('Extracting new features...')
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    
    gc.collect()
    train_df = do_count( train_df, ['ip', 'day', 'hour'], 'ip_tcount', 'uint16',show_max=True ); gc.collect()
    train_df = do_count( train_df, ['ip'], 'ip_count', show_max=True ); gc.collect()
    train_df = do_count( train_df, ['ip', 'device'], 'extra2', show_max=True ); gc.collect()
    train_df = do_count( train_df, ['ip','device','os'],'X0', show_max=True ); gc.collect()
    train_df = do_count( train_df, ['app', 'channel'],'X1', show_max=True ); gc.collect()
    train_df = do_count( train_df, ['ip','day'],'X2',show_max=True ); gc.collect()
    
    train_df = do_countuniq( train_df, ['ip'], 'channel', 'U1', show_max=True ); gc.collect()
    train_df = do_countuniq( train_df, ['ip', 'device', 'os'], 'app', 'U2',show_max=True ); gc.collect()
    train_df = do_countuniq( train_df, ['ip'], 'os', 'U3', show_max=True ); gc.collect()
    train_df = do_countuniq( train_df, ['app','channel'], 'os', 'U4', show_max=True ); gc.collect()
    train_df = do_countuniq( train_df, ['ip'], 'app', 'U5', show_max=True ); gc.collect()
    train_df = do_countuniq( train_df, ['ip','hour'], 'channel', 'U6', show_max=True ); gc.collect()
    
    del train_df['day']
    gc.collect()
    
    train_df = add_next_click(train_df); gc.collect()
    train_df = add_next_click2(train_df); gc.collect()
    train_df = add_next_click3(train_df); gc.collect()
    train_df = add_next_click4(train_df); gc.collect()
    train_df = add_next_click5(train_df); gc.collect()

    print('Extracting history clicks features...')
    HISTORY_CLICKS = {
        'identical_clicks': ['app','channel'],
        'app_clicks': ['ip', 'app'],
        'new_clicks': ['ip', 'device', 'os'],
    }
    
    # Go through different group-by combinations
    for fname, fset in HISTORY_CLICKS.items():
        # Clicks in the future
        train_df['future_'+fname] = train_df.iloc[::-1]. \
            groupby(fset). \
            cumcount(). \
            rename('future_'+fname).iloc[::-1]
        train_df['future_'+fname]=train_df['future_'+fname].astype('uint32')
    
    print("vars and data type: ")
    train_df.info()
    
    target = 'is_attributed'
    predictors.extend(['app','device','os', 'channel', 'hour', 
                  'ip_tcount', 'ip_count','extra2','X0','X1','X2',
                  'U1','U2','U3', 'U4','U5','U6',
                   'future_identical_clicks', 'future_app_clicks','future_new_clicks',
                   'next_click', 'next_click2', 'next_click3', 'next_click4', 'next_click5'])
    categorical = ['app', 'device', 'os', 'channel', 'hour']
    print('predictors',predictors)

    test_df = train_df[len_train:]
    val_df = train_df[(len_train-val_size):len_train]
    train_df = train_df[:(len_train-val_size)]

    print("train size: ", len(train_df))
    print("valid size: ", len(val_df))
    print("test size : ", len(test_df))

    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')

    gc.collect()

    print("Training...")
    start_time = time.time()

    params = {
        'learning_rate': 0.20,
        #'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 7,  # 2^max_depth - 1
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight':200 # because training data is extremely unbalanced 
    }
    (bst,best_iteration) = lgb_modelfit_nocv(params, 
                            train_df, 
                            val_df, 
                            predictors, 
                            target, 
                            objective='binary', 
                            metrics='auc',
                            early_stopping_rounds=30, 
                            verbose_eval=True, 
                            num_boost_round=1000, 
                            categorical_features=categorical)

    print('[{}]: model training time'.format(time.time() - start_time))
    del train_df
    del val_df
    gc.collect()

    print("Predicting...")
    sub['is_attributed'] = bst.predict(test_df[predictors],num_iteration=best_iteration)
    if not debug:
        print("writing...")
        sub.to_csv('D:/machine-learning/kaggle/talkingdata/sub_it%d.csv'%(fileno),index=False,float_format='%.9f')
    print("done...")
    return sub

nrows=184903891-1
nchunk=175595322
val_size=2500000

frm=nrows-175595322
if debug:
    frm=0
    nchunk=100000
    val_size=10000

to=frm+nchunk


sub=DO(frm,to,4)