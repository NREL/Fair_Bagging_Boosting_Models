import os
import numpy as np 
import pandas as pd 
import argparse 
import warnings
from tools.bias_utils import add_demographic_data, to_dmatrix, logit, inv_logit, r2_score
from tools.loss_functions import get_distance_corrected_mse, get_pearson_corrected_mse, get_kendalls_corrected_mse
from tools.optimizers import XGBOpt, RFOpt, GBTOpt
import json
warnings.filterwarnings("ignore")
# add directory to path
import sys
import xgboost as xgb
from xgb_wrappers.gradient_boosted_trees import GradientBoostedTreesModel
from xgb_wrappers.random_forest import RandomForestModel
sys.path.append('./tools/compiled_loss_funcs')

# Setup parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', '-m', type=str, default='xgb', help='Model type to use for training (xgb, rf, or gbt)')
parser.add_argument('--correction_type', '-c', type=str, default='pearson', help='Correction type to use for training (pearson, kendall, or distance)')
parser.add_argument('--demographic', '-d', type=str, default='MINRTY', help='Demographic to use for training')
parser.add_argument('--troubleshooting', '-t', action='store_true', help='Flag to run troubleshooting')
args = parser.parse_args()
n_nodes = -1

loss_dict = {'pearson': get_pearson_corrected_mse, 'kendall': get_kendalls_corrected_mse, 'distance': get_distance_corrected_mse}
opt_dict = {'xgb': XGBOpt, 'rf': RFOpt, 'gbt': GBTOpt}

def os_print(string):
    os.system('echo ' + f'Model: {args.model_type}, Correction: {args.correction_type}, ' + repr(string))

def train_model(params, dtrain, model_type, correction_type, gamma):
    objective = loss_dict[correction_type](gamma, etype=0)
    if model_type=='xgb':
        n_estimators = 100*int(params['n'])
        lr = 10**params['p']
        depth = int(params['depth'])
        gamma = 10**params['g']
        subsample = params['subsample']
        colsample = params['colsample']
        min_cw = params['min_cw']
        params = {'learning_rate':lr, 'max_depth':depth, 'n_jobs':-1, 'gamma':gamma, 'min_child_weight':min_cw,
            'subsample':subsample, 'colsample_bytree':colsample, 'random_state':1234}
        model = xgb.train(params, dtrain, num_boost_round=n_estimators, obj=objective)
    elif model_type == 'rf':
        n_estimators = 100*int(params['n'])
        depth = int(params['depth'])
        gamma = 10**params['g']
        min_cw = params['min_cw']
        subsample = params['subsample']
        colsample = params['colsample']
        params = {'n_estimators':n_estimators, 'max_depth':depth, 'nthread':-1, 'gamma':gamma, 'min_child_weight':min_cw,
                    'subsample':subsample, 'colsample_bytree':colsample, 'random_state':1234}
        model = RandomForestModel(**params)
        model.train(dtrain, obj=objective)
    elif model_type == 'gbt':
        n_estimators = 100*int(params['n'])
        lr = 10**params['p']
        depth = int(params['depth'])
        min_cw = params['min_cw']
        subsample = params['subsample']
        colsample = params['colsample']
        params = {'num_boost_round':n_estimators, 'eta':lr, 'max_depth':depth, 'nthread':-1, 'min_child_weight':min_cw,
                            'subsample':subsample, 'colsample_bytree':colsample, 'random_state':1234}
        model = GradientBoostedTreesModel(**params)
        model.train(dtrain, obj=objective)
    else:
        raise ValueError('Model type not recognized')

    return model

def main():
    demographic = args.demographic
    path = os.path.join(f'results_{args.model_type}', args.correction_type, demographic)
    if not os.path.exists(path):
        os.makedirs(path)

    # Load data
    os_print('Loading data...')
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
    X_train = X_train.drop(['fold', 'StationId'], axis = 1).values
    y_train = y.loc[X.fold!=0, :].values
    dtrain = to_dmatrix(X_train, y_train)

    X_test = X.loc[X.fold==0, :]
    dem_test = X_test[[demographic]].values.flatten()
    X_test = X_test.drop(['fold', 'StationId'], axis = 1).values
    y_test = y.loc[X.fold==0, :].values.flatten()
    dtest = to_dmatrix(X_test, y_test)

    # Optimize hyperparameters
    os_print('Sweeping Gamma...')
    params_path = os.path.join('.', 'base_models', f'{args.model_type}', 'params.json')
    with open(params_path) as f:
        params = json.load(f)
    n = 1 if args.troubleshooting else 50
    gammas = inv_logit(np.linspace(logit(0.5), logit(0.9999), n))
    gammas = np.concatenate((gammas, np.array([0.0])))
    results = pd.DataFrame({'gamma':gammas, 'test_mae':np.zeros(len(gammas)), 'test_r2':np.zeros(len(gammas)),
                            'test_rmse':np.zeros(len(gammas)), 'test_mape':np.zeros(len(gammas)),
                            'r2_0 0.6':np.zeros(len(gammas)), 'r2_0 0.7':np.zeros(len(gammas)), 'r2_0 0.8':np.zeros(len(gammas)),
                            'r2_0 0.9':np.zeros(len(gammas)), 'r2_1 0.6':np.zeros(len(gammas)), 'r2_1 0.7':np.zeros(len(gammas)), 
                            'r2_1 0.8':np.zeros(len(gammas)), 'r2_1 0.9':np.zeros(len(gammas)), 'r2 diff 0.6':np.zeros(len(gammas)),
                            'r2 diff 0.7':np.zeros(len(gammas)), 'r2 diff 0.8':np.zeros(len(gammas)), 'r2 diff 0.9':np.zeros(len(gammas))})
    if not os.path.exists(os.path.join(path, 'models')):
        os.makedirs(os.path.join(path, 'models'))

    for i in range(len(gammas)):
        gamma = gammas[i]
        os_print(f'Gamma: {gamma}, {i+1}/{len(gammas)}')
        model = train_model(params, dtrain, args.model_type, args.correction_type, gamma)
        model.save_model(os.path.join(path, 'models', f'{gamma}.model'))
        preds = model.predict(dtest).flatten()
        results.loc[i, 'test_mae'] = np.mean(np.abs(preds - y_test))
        results.loc[i, 'test_r2'] = r2_score(y_test, preds)
        results.loc[i, 'test_rmse'] = np.sqrt(np.mean((preds - y_test)**2))
        results.loc[i, 'test_mape'] = np.mean(np.abs(preds - y_test)/y_test)
        for c in [0.6, 0.7, 0.8, 0.9]:
            inds = dem_test < c 
            r2_0 = r2_score(y_test[inds], preds[inds])
            r2_1 = r2_score(y_test[~inds], preds[~inds])
            results.loc[i, f'r2_0 {c}'] = r2_0
            results.loc[i, f'r2_1 {c}'] = r2_1
            results.loc[i, f'r2 diff {c}'] = abs(r2_1 - r2_0)
    
    results.to_csv(os.path.join(path, 'gamma_sweep.csv'), index=False)

    
if __name__ == "__main__":
    main()
    os_print('Done!')