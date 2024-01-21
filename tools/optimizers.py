from bayes_opt import BayesianOptimization
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import json
from scipy import stats
from statsmodels.stats.dist_dependence_measures import distance_correlation
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
from tools.loss_functions import get_pearson_corrected_mse, get_distance_corrected_mse, get_kendalls_corrected_mse
from tools.bias_utils import to_dmatrix, inv_logit, logit

correction_dict = {'pearson':get_pearson_corrected_mse, 'distance':get_distance_corrected_mse, 'kendall':get_kendalls_corrected_mse}

def pearson_correlation_penalty(y_train, y_pred, dems, etype, gamma):
    y_train = y_train.reshape(np.shape(y_pred))
    diffs = y_pred - y_train #predt is the predicted y values
                                        # .get_label() returns the true y values
    rawvals = diffs # Raw residual
    if etype == 0:
        diffs = rawvals
    elif etype == 1:
        diffs = np.abs(diffs)
    else:
        diffs = diffs**2 # Squared residual

    n = len(diffs)
    mu_x = np.mean(diffs)
    var_x = np.sum((diffs-mu_x)**2) # Denominator cancels out
    mu_d = np.mean(dems)
    cov = np.sum((diffs-mu_x)*(dems-mu_d)) # Denominator cancels out
    var_d = np.sum((dems-mu_d)**2) # Denominator cancels out
    r2 = cov**2/(var_x*var_d)
    penalty = r2 # r2_score(diffs, dems.reshape(np.shape(diffs)))
    return penalty

def kendall_correlation_penalty(y_train, y_pred, dems, etype, gamma):
    y_train = y_train.reshape(np.shape(y_pred))
    diffs = y_pred - y_train #predt is the predicted y values
                                        # .get_label() returns the true y values
        

    rawvals = diffs # Raw residual
    if etype == 0:
        diffs = rawvals
    elif etype == 1:
        diffs = np.abs(diffs)
    else:
        diffs = diffs**2 # Squared residual
        
    
    penalty = stats.kendalltau(diffs, dems)[0]**2
    return penalty

def distance_correlation_penalty(y_train, y_pred, dems, etype, gamma):
    y_train = y_train.reshape(np.shape(y_pred))
    diffs = y_pred - y_train #predt is the predicted y values
                                        # .get_label() returns the true y values
    rawvals = diffs # Raw residual
    if etype == 0:
        diffs = rawvals
    elif etype == 1:
        diffs = np.abs(diffs)
    else:
        diffs = diffs**2 # Squared residual
    # Convert diffs to float64
    diffs = diffs.astype(np.float64)
    penalty = distance_correlation(diffs, dems)
    return penalty

penalty_dict = {'pearson':pearson_correlation_penalty, 'distance':distance_correlation_penalty, 'kendall':kendall_correlation_penalty}

class XGBOpt:
    def __init__(self, X_train, y_train, fold_train, dems, correction_type='pearson', n_nodes=-1):
        self.X_train = X_train
        self.y_train = y_train
        self.fold_train = fold_train
        self.dems = dems
        self.n_nodes = n_nodes
        self.get_correction = correction_dict[correction_type]
        self.correction_type = correction_type
        self.penalty = penalty_dict[correction_type]
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

    def xgboostcv(self, X, y, fold, n_estimators, lr, depth, n_jobs, gamma, min_cw, subsample, colsample, correction_gamma=None):
        uid = np.unique(fold)
        model_pred = np.zeros(X.shape[0])
        model_valid_loss = np.zeros(len(uid))
        model_train_loss = np.zeros(len(uid))
        cg = 0 if correction_gamma is None else inv_logit(correction_gamma)
        # comp = False if self.correction_type =='kendall' else True
        if (cg != None) and (cg != 0):
            objective = self.get_correction(cg, etype=0)
        else:
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

    def xgboostcv_mse(self, n, p, depth, g, min_cw, subsample, colsample, correction_gamma=None):
        model_cv = self.xgboostcv(self.X_train, self.y_train, self.fold_train, 
                            int(n)*100, 10**p, int(depth), self.n_nodes, 
                            10**g, min_cw, subsample, colsample, correction_gamma) # This CV training uses the correction_gamma
        MSE = mean_squared_error(self.y_train, model_cv['pred'])
        return -MSE
    
    def optimize(self, n_iter=50, init_points=10):
        self.optimizer.maximize(n_iter=n_iter, init_points=init_points)
        self.params = self.optimizer.max['params']
        self.best_mse = -self.optimizer.max['target']

    def optimize_gamma(self, n_iter=50, init_points=10, ratio_limit=1.1):
        if self.params is None:
            self.optimize(n_iter=n_iter, init_points=init_points)
        params = self.params
        
        def gamma_xgboostcv(n=params['n'], p=params['p'], depth=params['depth'], g=params['g'], min_cw=params['min_cw'], subsample=params['subsample'], colsample=params['colsample'], correction_gamma=None):
            etype = 0
            model_cv = self.xgboostcv(self.X_train, self.y_train, self.fold_train, 
                                int(n)*100, 10**p, int(depth), self.n_nodes, 
                                10**g, min_cw, subsample, colsample, correction_gamma) # This CV training uses the correction_gamma
            pred_y = model_cv['pred']
            MSE = mean_squared_error(self.y_train, pred_y)
            etype = round(etype)  # In case etype is given as float
            penalty = self.penalty(self.y_train, pred_y, self.dems, etype, correction_gamma)
            
            # We don't want MSE to be worse than 10% or rmse without gamma
            ratio = np.sqrt(MSE/self.best_mse)
            print('rmse ratio: ', ratio, 'r2_penalty: ', penalty)
            mse_penalty = 0
            
            # We tolerate up to 10% increase in MSE compared to the MSE from the uncorrected model
            if ratio <= ratio_limit:
                mse_penalty = penalty
            else: 
                mse_penalty = 1 + penalty
            return -mse_penalty
        
        self.gamma_optimizer = BayesianOptimization(gamma_xgboostcv,{
                                                    'correction_gamma': (logit(0.5), logit(1-1e-3))
                                                    })
        self.gamma_optimizer.maximize(n_iter=n_iter, init_points=init_points)
        self.gamma_params = self.gamma_optimizer.max['params']

    def save_results(self, save_folder, save_gamma=True):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        with open(os.path.join(save_folder, 'params.json'), 'w') as f:
            json.dump(self.params, f)
        if save_gamma:
            with open(os.path.join(save_folder, 'gamma_params.json'), 'w') as f:
                json.dump(self.gamma_params, f)
            with open(os.path.join(save_folder, 'best_mse.json'), 'w') as f:
                json.dump(self.best_mse, f)

    def run(self, save_folder, n_iter=50, init_points=10, ratio_limit=1.1):
        self.optimize(n_iter=n_iter, init_points=init_points)
        self.optimize_gamma(n_iter=n_iter, init_points=init_points, ratio_limit=ratio_limit)
        self.save_results(save_folder)

    def run_base(self, save_folder, n_iter=50, init_points=10):
        self.optimize(n_iter=n_iter, init_points=init_points)
        self.save_results(save_folder, save_gamma=False)

    def run_gamma(self, params, save_folder, n_iter=50, init_points=10, ratio_limit=1.1):
        self.params = params
        self.best_mse = -self.xgboostcv_mse(**params)
        self.optimize_gamma(n_iter=n_iter, init_points=init_points, ratio_limit=ratio_limit)
        self.save_results(save_folder, save_gamma=True)


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

    def xgboostcv(self, X, y, fold, n_estimators, depth, n_jobs, gamma, min_cw, subsample, colsample, correction_gamma=None):
        uid = np.unique(fold)
        model_pred = np.zeros(X.shape[0])
        model_valid_loss = np.zeros(len(uid))
        model_train_loss = np.zeros(len(uid))
        cg = 0 if correction_gamma is None else inv_logit(correction_gamma)
        # comp = False if self.correction_type =='kendall' else True
        if (cg != None) and (cg != 0):
            objective = self.get_correction(cg, etype=0)
        else:
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

    def xgboostcv_mse(self, n, depth, g, min_cw, subsample, colsample, correction_gamma=None):
        model_cv = self.xgboostcv(self.X_train, self.y_train, self.fold_train, 
                            int(n)*100, int(depth), self.n_nodes, 
                            10**g, min_cw, subsample, colsample, correction_gamma) # This CV training uses the correction_gamma
        MSE = mean_squared_error(self.y_train, model_cv['pred'])
        return -MSE

    def optimize_gamma(self, n_iter=50, init_points=10, ratio_limit=1.1):
        if self.params is None:
            self.optimize(n_iter=n_iter, init_points=init_points)
        params = self.params
        
        def gamma_xgboostcv(n=params['n'], depth=params['depth'], g=params['g'], min_cw=params['min_cw'], subsample=params['subsample'], colsample=params['colsample'], correction_gamma=None):
            etype = 0
            model_cv = self.xgboostcv(self.X_train, self.y_train, self.fold_train, 
                                int(n)*100, int(depth), self.n_nodes, 
                                10**g, min_cw, subsample, colsample, correction_gamma) # This CV training uses the correction_gamma
            pred_y = model_cv['pred']
            MSE = mean_squared_error(self.y_train, pred_y)
            etype = round(etype)  # In case etype is given as flot
            penalty = self.penalty(self.y_train, pred_y, self.dems, etype, correction_gamma)
            
            # We don't want MSE to be worse than 10% or rmse without gamma
            ratio = np.sqrt(MSE/self.best_mse)
            print('rmse ratio: ', ratio, 'r2_penalty: ', penalty)
            mse_penalty = 0
            
            # We tolerate up to 10% increase in MSE compared to the MSE from the uncorrected model
            if ratio <= ratio_limit:
                mse_penalty = penalty
            else: 
                mse_penalty = 1 + penalty
            return -mse_penalty
        
        self.gamma_optimizer = BayesianOptimization(gamma_xgboostcv,{
                                                    'correction_gamma': (logit(0.5), logit(1-1e-3))
                                                    })
        self.gamma_optimizer.maximize(n_iter=n_iter, init_points=init_points)
        self.gamma_params = self.gamma_optimizer.max['params']
    



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

    def xgboostcv(self, X, y, fold, n_estimators, lr, depth, n_jobs, min_cw, subsample, colsample, correction_gamma=None):
        uid = np.unique(fold)
        model_pred = np.zeros(X.shape[0])
        model_valid_loss = np.zeros(len(uid))
        model_train_loss = np.zeros(len(uid))
        cg = 0 if correction_gamma is None else inv_logit(correction_gamma)
        # comp = False if self.correction_type =='kendall' else True
        if (cg != None) and (cg != 0):
            objective = self.get_correction(cg, etype=0)
        else:
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

    def optimize_gamma(self, n_iter=50, init_points=10, ratio_limit=1.1):
        if self.params is None:
            self.optimize(n_iter=n_iter, init_points=init_points)
        params = self.params
        
        def gamma_xgboostcv(n=params['n'], p=params['p'], depth=params['depth'], min_cw=params['min_cw'], subsample=params['subsample'], colsample=params['colsample'], correction_gamma=None):
            etype = 0
            model_cv = self.xgboostcv(self.X_train, self.y_train, self.fold_train, 
                                int(n)*100, 10**p, int(depth), self.n_nodes, min_cw, subsample, colsample, correction_gamma) # This CV training uses the correction_gamma
            pred_y = model_cv['pred']
            MSE = mean_squared_error(self.y_train, pred_y)
            etype = round(etype)  # In case etype is given as flot
            penalty = self.penalty(self.y_train, pred_y, self.dems, etype, correction_gamma)
            
            # We don't want MSE to be worse than 10% or rmse without gamma
            ratio = np.sqrt(MSE/self.best_mse)
            print('rmse ratio: ', ratio, 'r2_penalty: ', penalty)
            mse_penalty = 0
            
            # We tolerate up to 10% increase in MSE compared to the MSE from the uncorrected model
            if ratio <= ratio_limit:
                mse_penalty = penalty
            else: 
                mse_penalty = 1 + penalty
            return -mse_penalty
        
        self.gamma_optimizer = BayesianOptimization(gamma_xgboostcv,{
                                                    'correction_gamma': (logit(0.5), logit(1-1e-3))
                                                    })
        self.gamma_optimizer.maximize(n_iter=n_iter, init_points=init_points)
        self.gamma_params = self.gamma_optimizer.max['params']