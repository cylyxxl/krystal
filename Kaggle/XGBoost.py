import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm
import random
import seaborn as sns
import math
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import xgboost as xgb
import gc

DATA_PATH = Path('../input/ump-train-picklefile')
SAMPLE_TEST_PATH = Path('../input/ubiquant-market-prediction')

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

train = pd.read_pickle(DATA_PATH/'train.pkl')
train = reduce_mem_usage(train)

train.drop(['row_id', 'time_id'], axis=1, inplace=True)
X = train.drop(['target'], axis=1)
y = train["target"]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.01, random_state=42, shuffle=False)

model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=12,
    subsample=0.9,
    colsample_bytree=0.7,
    missing=-999,
    random_state=1111,
    tree_method='gpu_hist'  
    )
    
model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_valid, y_valid)], verbose=1)

import ubiquant
env = ubiquant.make_env()  
iter_test = env.iter_test()
for (test_df, sample_prediction_df) in iter_test:
    test_df.drop(['row_id'], axis=1, inplace=True)
    pred = model.predict(test_df)
    sample_prediction_df['target'] = pred
    env.predict(sample_prediction_df) 