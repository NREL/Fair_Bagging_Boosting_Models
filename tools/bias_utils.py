import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import DMatrix

def add_demographic_data(data, demo, dropid=False, loc='./data/off_freeway_station_demographics.csv'):
    if isinstance(demo, str):
        demo = [demo]
    demos = pd.read_csv(loc)
    cols = ['STN_NUMBER'] +['EPL_' + d for d in demo]
    demos = demos[cols]
    # rename columns
    dct = {'EPL_'+d:d for d in demo}
    demos.rename(columns=dct, inplace=True)
    demos.rename(columns={'STN_NUMBER':'StationId'}, inplace=True)  
    # drop demos not in data
    demos = demos[demos['StationId'].isin(data['StationId'].unique())]
    data = pd.merge(data, demos, on='StationId', how='left')
    if dropid:
        data.drop('StationId', axis=1, inplace=True)
    return data

def to_dmatrix(X, y):
    # get number of columns of np array X
    n_cols = X.shape[1]
    weights = [1.0 for _ in range(n_cols-1)] + [0.0]
    return DMatrix(X, label=y, feature_weights=weights)

def get_col_data(data, col):
    return data[['Volume', 'PredVolume', col]]
def get_s(r2_0, r2_1):
    # check if r2 is a ndarray
    if isinstance(r2_0, np.ndarray):
        r2_0 = r2_0[0]
        r2_1 = r2_1[0]
    elif isinstance(r2_0, pd.Series):
        r2_0 = r2_0.iloc[0]
        r2_1 = r2_1.iloc[0]
    if r2_0 > r2_1:
        return r2_0 / r2_1
    else:
        return r2_1 / r2_0

def get_stats(col, col_data):
    col_data_0 = col_data[col_data[col] == 0]
    col_data_1 = col_data[col_data[col] == 1]
    col_r2_0 = np.corrcoef(col_data_0['Volume'], col_data_0['PredVolume'])[0, 1]**2
    col_r2_1 = np.corrcoef(col_data_1['Volume'], col_data_1['PredVolume'])[0, 1]**2
    stat = get_s(col_r2_0, col_r2_1)
    n0 = sum(col_data[col] == 0)
    n1 = sum(col_data[col] == 1)
    return stat, n0, n1, col_r2_0, col_r2_1

def r2_resample(n, col_data):
    sample = col_data.sample(n, replace=True)
    r2 = np.corrcoef(sample['Volume'], sample['PredVolume'])[0,1]**2
    return r2

def stat_resample(n0, n1, col_data):
    r2_0 = r2_resample(n0, col_data)
    r2_1 = r2_resample(n1, col_data)
    return max([r2_0/r2_1, r2_1/r2_0])

def bootstrap(n0, n1, col_data, b=10000):
    stat = []
    for i in range(b):
        stat.append(stat_resample(n0, n1, col_data))
    return np.array(stat)

def get_r2_pval(data, col):
    col_data = get_col_data(data, col)
    stat, n0, n1, r2_0, r2_1 = get_stats(col, col_data)
    stat_boot = bootstrap(n0, n1, col_data)
    pval = sum(stat_boot > stat)/len(stat_boot)
    return r2_0, r2_1, pval