import os
import numpy as np 
import pandas as pd 
import warnings
from tools.test_bias import test_bias
from tools.plotting import plot_bias
from tools.bias_utils import add_demographic_data, to_dmatrix, logit, inv_logit
from tools.loss_functions import get_distance_corrected_mse, get_pearson_corrected_mse, get_kendalls_corrected_mse
from xgb_wrappers.gradient_boosted_trees import GradientBoostedTreesModel
from xgb_wrappers.random_forest import RandomForestModel
import json
import xgboost as xgb
import time
warnings.filterwarnings("ignore")
# add directory to path
import sys
sys.path.append('./tools/compiled_loss_funcs')

model_dict = {'rf': RandomForestModel, 'gbt': GradientBoostedTreesModel}
loss_dict = {'pearson': get_pearson_corrected_mse, 'kendall': get_kendalls_corrected_mse, 'distance': get_distance_corrected_mse}

def process(cutoff, correction, model_type, demographic='MINRTY', train=False):
    path = os.path.join('results', f'results_{cutoff}', model_type, correction, demographic)
    # Load params
    with open(os.path.join(path, 'params.json')) as f:
        params = json.load(f)
    with open(os.path.join(path, 'gamma_params.json')) as f:
        correction_gamma = inv_logit(json.load(f)["correction_gamma"])

    cols_drop = ['Date', 'FC', 'PenRate', 'NumberOfLanes', 'Dir', 'Lat', 'Long']
    raw_data_train = pd.read_csv("./data/final_train_data.csv")
    raw_data_test = pd.read_csv("./data/final_test_data.csv")
    raw_data_test1 = pd.DataFrame(np.concatenate((raw_data_test.values, np.zeros(raw_data_test.shape[0]).reshape(-1, 1)), axis=1),
                                    columns = raw_data_test.columns.append(pd.Index(['fold'])))
    raw_data = pd.DataFrame(np.concatenate((raw_data_train.values, raw_data_test1.values), axis=0), 
                            columns = raw_data_train.columns)

    raw_data = add_demographic_data(raw_data, demographic)
    raw_data = raw_data.dropna()
    raw_data_train = raw_data.loc[raw_data.fold!=0, :]
    raw_data_test = raw_data.loc[raw_data.fold==0, :]
    data = raw_data.drop(cols_drop, axis=1)
    if 'Dir' in data.columns:
        one_hot = pd.get_dummies(data[['Dir']])
        data = data.drop(['Dir'], axis = 1)
        data = data.join(one_hot)
    week_dict = {"DayOfWeek": {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 
                                'Friday': 5, 'Saturday': 6, 'Sunday': 7}}
    data = data.replace(week_dict)

    X = data.drop(['Volume'], axis=1)
    y = data[['Volume']]

    X_train = X.loc[X.fold!=0, :]
    dem_train = X_train[[demographic]].values
    fold_train = X_train[['fold']].values.reshape(-1)
    train_ids = X_train[['StationId']].values.reshape((len(X_train), 1))
    X_col = X_train.drop(['fold', 'StationId'], axis = 1).columns
    X_train = X_train.drop(['fold', 'StationId'], axis = 1).values
    y_train = y.loc[X.fold!=0, :].values

    X_test = X.loc[X.fold==0, :]
    dem_test = X_test[[demographic]].values
    test_ids = X_test[['StationId']].values.reshape((len(X_test), 1))
    X_test = X_test.drop(['fold', 'StationId'], axis = 1).values
    y_test = y.loc[X.fold==0, :].values

    dtrain = to_dmatrix(X_train, y_train)
    dtest = to_dmatrix(X_test, y_test)

    n_jobs = -1
    objective = loss_dict[correction](correction_gamma, etype=0)
    objective_uncorrected = get_pearson_corrected_mse(0.0, etype=0)
    corrected_path = os.path.join(path, 'corrected.model')
    if model_type == 'xgb':
        base_name = 'base_xgb.model'
    elif model_type == 'rf':
        base_name = 'base_rf.model'
    elif model_type == 'gbt':
        base_name = 'base_gbt.model'

    base_path = os.path.join('.', 'base_models', model_type, base_name)
    if not os.path.exists(corrected_path) or not os.path.exists(base_path) or train:
        print('No model found, retraining...')
        train = True # Force training if no saved model

    if model_type == 'xgb':
        n_estimators = 100*int(params['n'])
        lr = 10**params['p']
        depth = int(params['depth'])
        gamma = 10**params['g']
        subsample = params['subsample']
        colsample = params['colsample']
        min_cw = params['min_cw']
        params = {'learning_rate':lr, 'max_depth':depth, 'n_jobs':n_jobs, 'gamma':gamma, 'min_child_weight':min_cw,
            'subsample':subsample, 'colsample_bytree':colsample, 'random_state':1234}
        model_uncorrected = xgb.Booster(params)
        model_uncorrected.load_model(base_path)
        if train:
            model_corrected = xgb.train(params, dtrain, num_boost_round=n_estimators, obj=objective)
            model_uncorrected = xgb.train(params, dtrain, num_boost_round=n_estimators, obj=objective_uncorrected)
            model_corrected.save_model(corrected_path)
        else:
            model_corrected = xgb.Booster(params)
            model_corrected.load_model(corrected_path)
    elif model_type == 'rf':
        n_estimators = 100*int(params['n'])
        depth = int(params['depth'])
        gamma = 10**params['g']
        min_cw = params['min_cw']
        subsample = params['subsample']
        colsample = params['colsample']
        params = {'n_estimators':n_estimators, 'max_depth':depth, 'nthread':n_jobs, 'gamma':gamma, 'min_child_weight':min_cw,
                    'subsample':subsample, 'colsample_bytree':colsample, 'random_state':1234}
        model_uncorrected = RandomForestModel(**params)
        model_uncorrected.load_model(base_path)
        if train:
            model_corrected = RandomForestModel(**params)
            model_corrected.train(dtrain, obj=objective)
            model_corrected.save_model(corrected_path)
        else:
            model_corrected = RandomForestModel(**params)
            model_corrected.load_model(corrected_path)
    elif model_type == 'gbt':
        n_estimators = 100*int(params['n'])
        lr = 10**params['p']
        depth = int(params['depth'])
        min_cw = params['min_cw']
        subsample = params['subsample']
        colsample = params['colsample']
        params = {'num_boost_round':n_estimators, 'eta':lr, 'max_depth':depth, 'nthread':n_jobs, 'min_child_weight':min_cw,
                            'subsample':subsample, 'colsample_bytree':colsample, 'random_state':1234}
        model_uncorrected = GradientBoostedTreesModel(**params)
        model_uncorrected.load_model(base_path)
        if train:
            model_corrected = GradientBoostedTreesModel(**params)
            model_corrected.train(dtrain, obj=objective)
            model_corrected.save_model(corrected_path)
        else:
            model_corrected = GradientBoostedTreesModel(**params)
            model_corrected.load_model(corrected_path)
    else:
        raise ValueError('Model type not recognized')

    print('Testing Bias')
    base_path = os.path.join('.', 'base_models', model_type, 'bias.csv')
    if os.path.exists(base_path):
        bias_data = pd.read_csv(base_path)
    else:
        bias_data = test_bias(model_uncorrected, dtest, dem_test)
        bias_data.to_csv(base_path, index=False)

    corrected_path = os.path.join(path, 'bias.csv')
    force = True
    if not force and os.path.exists(corrected_path):
        bias_data_corrected = pd.read_csv(corrected_path)
    else:
        bias_data_corrected = test_bias(model_corrected, dtest, dem_test)
        bias_data_corrected.to_csv(corrected_path, index=False)
    plot_data = pd.concat([bias_data, bias_data_corrected], axis=0).reset_index(drop=True)

    labels = ['Original', f'{correction} Corrected']
    ylim=(0, 1.2)
    plot_bias(plot_data, labels, savepath=os.path.join(path, 'bias_testing.jpg'), ylim=ylim, show=False)

def main():
    cutoffs = [0.5, 1.5]
    corrections = ['pearson', 'distance', 'kendall']
    model_types = ['xgb', 'rf', 'gbt']

    for correction in corrections:
        for model_type in model_types:
            for cutoff in cutoffs:
                try:
                    start = time.time()
                    print(f'Started processing {correction} correction with {model_type} model and cutoff {cutoff}')
                    process(cutoff, correction, model_type)
                    print(f'Processed {correction} correction with {model_type} model and cutoff {cutoff}')
                    print(f'Time elapsed: {time.time()-start}')
                except Exception as e:
                    print(f'Error processing {correction} correction with {model_type} model and cutoff {cutoff}')
                    print(e)

if __name__ == '__main__':
    main()