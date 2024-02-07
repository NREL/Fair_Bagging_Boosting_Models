import os
import numpy as np
import pandas as pd
import argparse
import warnings
from tools.bias_utils import add_demographic_data
from tools.loss_functions import get_pearson_corrected_mse
from tools.optimizers import XGBOpt, RFOpt, GBTOpt, to_dmatrix
import xgboost as xgb
from xgb_wrappers.gradient_boosted_trees import GradientBoostedTreesModel
from xgb_wrappers.random_forest import RandomForestModel
warnings.filterwarnings("ignore")
# add directory to path
import sys
sys.path.append('./tools/compiled_loss_funcs')

# Setup parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', '-m', type=str, default='xgb', help='Model type to use for training (xgb, rf, or gbt)')
parser.add_argument('--demographic', '-d', type=str, default='MINRTY', help='Demographic to test bias for')
parser.add_argument('--troubleshooting', '-t', action='store_true', help='Flag to run troubleshooting')
args = parser.parse_args()
n_nodes = -1

opt_dict = {'xgb': XGBOpt, 'rf': RFOpt, 'gbt': GBTOpt}

def os_print(string):
    os.system('echo ' + f'Model: {args.model_type}' + repr(string))

def main():
    demographic = args.demographic
    path = os.path.join(f'base_models/{args.model_type}/')
    if not os.path.exists(path):
        os.makedirs(path)

    # Load data
    os_print('Loading data...')
    cols_drop = ['Date', 'FC', 'PenRate', 'NumberOfLanes', 'Dir', 'Lat', 'Long']

    raw_data_train = pd.read_csv("./data/final_train_data_syn.csv")
    raw_data_test = pd.read_csv("./data/final_test_data_syn.csv")
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
    test_ids = X_test[['StationId']].values.reshape((len(X_test), 1))
    X_test = X_test.drop(['fold', 'StationId'], axis = 1).values
    y_test = y.loc[X.fold==0, :].values

    # Optimize hyperparameters
    os_print('Optimizing hyperparameters...')
    opt = opt_dict[args.model_type](X_train, y_train, fold_train, dem_train, n_nodes=n_nodes)
    init_points = 0 if args.troubleshooting else 10
    n_iter = 1 if args.troubleshooting else 50
    opt.run_base(path, init_points=init_points, n_iter=n_iter)
    params = opt.params

    # Train base model
    os_print('Training base model...')
    n_jobs=-1
    objective = get_pearson_corrected_mse(0.0, etype=0)
    dtrain = to_dmatrix(X_train, y_train)
    if args.model_type == 'xgb':
        n_estimators = 100*int(params['n'])
        lr = 10**params['p']
        depth = int(params['depth'])
        gamma = 10**params['g']
        subsample = params['subsample']
        colsample = params['colsample']
        min_cw = params['min_cw']
        params = {'learning_rate':lr, 'max_depth':depth, 'n_jobs':n_jobs, 'gamma':gamma, 'min_child_weight':min_cw,
            'subsample':subsample, 'colsample_bytree':colsample, 'random_state':1234}
        model = xgb.train(params, dtrain, num_boost_round=n_estimators, obj=objective)
        model.save_model(os.path.join(path, 'base_xgb.model'))
    elif args.model_type == 'rf':
        n_estimators = 100*int(params['n'])
        depth = int(params['depth'])
        gamma = 10**params['g']
        min_cw = params['min_cw']
        subsample = params['subsample']
        colsample = params['colsample']
        params = {'n_estimators':n_estimators, 'max_depth':depth, 'nthread':n_jobs, 'gamma':gamma, 'min_child_weight':min_cw,
                    'subsample':subsample, 'colsample_bytree':colsample, 'random_state':1234}
        model = RandomForestModel(**params)
        model.train(dtrain, obj=objective)
        model.save_model(os.path.join(path, 'base_rf.model'))
    elif args.model_type == 'gbt':
        n_estimators = 100*int(params['n'])
        lr = 10**params['p']
        depth = int(params['depth'])
        min_cw = params['min_cw']
        subsample = params['subsample']
        colsample = params['colsample']
        params = {'num_boost_round':n_estimators, 'eta':lr, 'max_depth':depth, 'nthread':n_jobs, 'min_child_weight':min_cw,
                            'subsample':subsample, 'colsample_bytree':colsample, 'random_state':1234}
        model = GradientBoostedTreesModel(**params)
        model.train(dtrain, obj=objective)
        model.save_model(os.path.join(path, 'base_gbt.model'))

if __name__ == "__main__":
    main()
    os_print('Done!')
