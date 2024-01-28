from bayes_opt import BayesianOptimization
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
import json
import warnings
warnings.filterwarnings("ignore")
# add parent directory to path
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from xgb_wrappers.gradient_boosted_trees import GradientBoostedTreesModel
from xgb_wrappers.random_forest import RandomForestModel
from tools.loss_functions import get_pearson_corrected_mse
from tools.bias_utils import to_dmatrix

class XGBOpt:
    def __init__(self, X_train, y_train, fold_train, dems, correction_type='pearson', n_nodes=-1):
        self.X_train = X_train
        self.y_train = y_train
        self.fold_train = fold_train
        self.dems = dems
        self.n_nodes = n_nodes
        self.correction_type = correction_type
        self.params = None
        self.setup_optimizer()

    def setup_optimizer(self):
         self.optimizer = BayesianOptimization(self.xgboostcv_mse,
                                 {'n': (1, 10),
                                  'p': (-4, 0),
                                  'depth': (2, 10),
                                  'g': (-3, 0),
                                  'min_cw': (1, 10), 
                                  'subsample': (0.5, 1), 
                                  'colsample': (0.5, 1), 
                                 })

    def xgboostcv(self, X, y, fold, n_estimators, lr, depth, n_jobs, gamma, min_cw, subsample, colsample):
        uid = np.unique(fold)
        model_pred = np.zeros(X.shape[0])
        model_valid_loss = np.zeros(len(uid))
        model_train_loss = np.zeros(len(uid))
        
        objective = get_pearson_corrected_mse(0, etype=0) # Pearson correction is fastest by far
        for i in uid:
            x_valid = X[fold==i]
            x_train = X[fold!=i]
            y_valid = y[fold==i]
            y_train = y[fold!=i]
            train_data = to_dmatrix(x_train, y_train)
            valid_data = to_dmatrix(x_valid, y_valid)
            params = {'learning_rate':lr, 'max_depth':depth, 'n_jobs':n_jobs, 'gamma':gamma, 'min_child_weight':min_cw,
                        'subsample':subsample, 'colsample_bytree':colsample, 'random_state':1234}
            model = xgb.train(params, train_data, num_boost_round=n_estimators, obj=objective)

            pred = model.predict(valid_data)
            model_pred[fold==i] = pred
            model_valid_loss[uid==i] = mean_squared_error(y_valid, pred)
            model_train_loss[uid==i] = mean_squared_error(y_train, model.predict(train_data))
        return {'pred':model_pred, 'valid_loss':model_valid_loss, 'train_loss':model_train_loss}

    def xgboostcv_mse(self, n, p, depth, g, min_cw, subsample, colsample):
        model_cv = self.xgboostcv(self.X_train, self.y_train, self.fold_train, 
                            int(n)*100, 10**p, int(depth), self.n_nodes, 
                            10**g, min_cw, subsample, colsample)
        MSE = mean_squared_error(self.y_train, model_cv['pred'])
        return -MSE
    
    def optimize(self, n_iter=50, init_points=10):
        self.optimizer.maximize(n_iter=n_iter, init_points=init_points)
        self.params = self.optimizer.max['params']
        self.best_mse = -self.optimizer.max['target']

    def save_results(self, save_folder):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        with open(os.path.join(save_folder, 'params.json'), 'w') as f:
            json.dump(self.params, f)

    def run_base(self, save_folder, n_iter=50, init_points=10):
        self.optimize(n_iter=n_iter, init_points=init_points)
        self.save_results(save_folder)



class RFOpt(XGBOpt):
    def setup_optimizer(self):
        self.optimizer = BayesianOptimization(self.xgboostcv_mse,
                                 {'n': (1, 10),
                                  'depth': (2, 10),
                                  'g': (-3, 0),
                                  'min_cw': (1, 10), 
                                  'subsample': (0.5, 0.999), 
                                  'colsample': (0.5, 0.999), 
                                 })

    def xgboostcv(self, X, y, fold, n_estimators, depth, n_jobs, gamma, min_cw, subsample, colsample):
        uid = np.unique(fold)
        model_pred = np.zeros(X.shape[0])
        model_valid_loss = np.zeros(len(uid))
        model_train_loss = np.zeros(len(uid))
        
        objective = get_pearson_corrected_mse(0, etype=0) # Pearson correction is fastest by far
        for i in uid:
            x_valid = X[fold==i]
            x_train = X[fold!=i]
            y_valid = y[fold==i]
            y_train = y[fold!=i]
            train_data = to_dmatrix(x_train, y_train)
            valid_data = to_dmatrix(x_valid, y_valid)
            params = {'n_estimators':n_estimators, 'max_depth':depth, 'nthread':n_jobs, 'gamma':gamma, 'min_child_weight':min_cw,
                        'subsample':subsample, 'colsample_bytree':colsample, 'random_state':1234}
            model = RandomForestModel(**params)
            model.train(train_data, obj=objective)

            pred = model.predict(valid_data)
            model_pred[fold==i] = pred
            model_valid_loss[uid==i] = mean_squared_error(y_valid, pred)
            model_train_loss[uid==i] = mean_squared_error(y_train, model.predict(train_data))
        return {'pred':model_pred, 'valid_loss':model_valid_loss, 'train_loss':model_train_loss}

    def xgboostcv_mse(self, n, depth, g, min_cw, subsample, colsample):
        model_cv = self.xgboostcv(self.X_train, self.y_train, self.fold_train, 
                            int(n)*100, int(depth), self.n_nodes, 
                            10**g, min_cw, subsample, colsample) # This CV training uses the correction_gamma
        MSE = mean_squared_error(self.y_train, model_cv['pred'])
        return -MSE
    

class GBTOpt(XGBOpt):
    def setup_optimizer(self):
        self.optimizer = BayesianOptimization(self.xgboostcv_mse,
                                 {'n': (1, 10),
                                  'p': (-4, 0),
                                  'depth': (2, 10),
                                  'min_cw': (1, 10), 
                                  'subsample': (0.5, 1), 
                                  'colsample': (0.5, 1), 
                                 })

    def xgboostcv(self, X, y, fold, n_estimators, lr, depth, n_jobs, min_cw, subsample, colsample):
        uid = np.unique(fold)
        model_pred = np.zeros(X.shape[0])
        model_valid_loss = np.zeros(len(uid))
        model_train_loss = np.zeros(len(uid))
        
        objective = get_pearson_corrected_mse(0, etype=0) # Pearson correction is fastest by far
        for i in uid:
            x_valid = X[fold==i]
            x_train = X[fold!=i]
            y_valid = y[fold==i]
            y_train = y[fold!=i]
            train_data = to_dmatrix(x_train, y_train)
            valid_data = to_dmatrix(x_valid, y_valid)
            params = {'num_boost_round':n_estimators, 'eta':lr, 'max_depth':depth, 'nthread':n_jobs, 'min_child_weight':min_cw,
                        'subsample':subsample, 'colsample_bytree':colsample, 'random_state':1234}
            model = GradientBoostedTreesModel(**params)
            model.train(train_data, obj=objective)

            pred = model.predict(valid_data)
            model_pred[fold==i] = pred
            model_valid_loss[uid==i] = mean_squared_error(y_valid, pred)
            model_train_loss[uid==i] = mean_squared_error(y_train, model.predict(train_data))
        return {'pred':model_pred, 'valid_loss':model_valid_loss, 'train_loss':model_train_loss}

    def xgboostcv_mse(self, n, p, depth, min_cw, subsample, colsample, correction_gamma=None):
        model_cv = self.xgboostcv(self.X_train, self.y_train, self.fold_train, 
                            int(n)*100, 10**p, int(depth), self.n_nodes, min_cw, subsample, colsample, correction_gamma) # This CV training uses the correction_gamma
        MSE = mean_squared_error(self.y_train, model_cv['pred'])
        return -MSE
