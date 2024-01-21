import os
import numpy as np 
import pandas as pd 
import argparse 
import warnings
from tools.test_bias import test_bias, train_test_bias
from tools.plotting import plot_bias
from tools.bias_utils import add_demographic_data
from tools.loss_functions import get_distance_corrected_mse, get_pearson_corrected_mse, get_kendalls_corrected_mse
from tools.optimizers import XGBOpt, RFOpt, GBTOpt
import json
warnings.filterwarnings("ignore")
# add directory to path
import sys
sys.path.append('./tools/compiled_loss_funcs')

# Setup parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', '-m', type=str, default='xgb', help='Model type to use for training (xgb, rf, or gbt)')
parser.add_argument('--correction_type', '-c', type=str, default='pearson', help='Correction type to use for training (pearson, kendall, or distance)')
parser.add_argument('--demographic', '-d', type=str, default='MINRTY', help='Demographic to use for training')
parser.add_argument('--troubleshooting', '-t', action='store_true', help='Flag to run troubleshooting')
parser.add_argument('--leeway', '-l', type=float, default=0.1, help='Leeway for rmse')
args = parser.parse_args()
n_nodes = -1

loss_dict = {'pearson': get_pearson_corrected_mse, 'kendall': get_kendalls_corrected_mse, 'distance': get_distance_corrected_mse}
opt_dict = {'xgb': XGBOpt, 'rf': RFOpt, 'gbt': GBTOpt}

def os_print(string):
    os.system('echo ' + f'Model: {args.model_type}, Correction: {args.correction_type}, ' + repr(string))

def main():
    demographic = args.demographic
    path = os.path.join(f'results_{args.leeway}', args.model_type, args.correction_type, demographic)
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
    opt = opt_dict[args.model_type](X_train, y_train, fold_train, dem_train, correction_type=args.correction_type, n_nodes=n_nodes)
    init_points = 0 if args.troubleshooting else 10
    n_iter = 1 if args.troubleshooting else 50
    params_path = os.path.join('.', 'base_models', f'{args.model_type}', 'params.json')
    with open(params_path) as f:
        params = json.load(f)

    # os_print(str(params))
    opt.run_gamma(params, path, init_points=init_points, n_iter=n_iter, ratio_limit=1+args.leeway)
    
if __name__ == "__main__":
    main()
    os_print('Done!')