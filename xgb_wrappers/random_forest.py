#######################################################################################################
#######################################################################################################
# Random Forest Model defined within XGBoost framework
# Created by Erik Bensen, National Renewable Energy Laboratory
# 5/10/2023
#######################################################################################################
#######################################################################################################

import xgboost as xgb
import numpy as np 
import pandas as pd

class RandomForestModel():
    def __init__(self, objective='reg:squarederror', verbosity=1, subsample=0.8, colsample_bytree=0.8,
                 colsample_bylevel=1.0, colsample_bynode=1.0, reg_lambda=1e-5, n_estimators=100, *, eval_metric=None,
                 validate_parameters=None, nthread=None, disable_default_eval_metric=None,
                 gamma=None, max_depth=None, min_child_weight=None, max_delta_step=None,
                 sampling_method=None, reg_alpha=None, tree_method=None, grow_policy=None,
                 max_leaves=None, random_state=None, seed_per_iteration=None):
        
        # Ensure parameters are valid for random forest
        if not (subsample < 1.0):
            raise ValueError('Subsample must be less than 1.0')
        if not (colsample_bylevel*colsample_bynode*colsample_bytree < 1.0):
            raise ValueError('At least one of the colsample parameters must be less than 1.0')
        
        self.params = {'objective':objective, 'booster':'gbtree', 'eta':1, 'verbosity':verbosity,
                       'subsample':subsample, 'colsample_bytree':colsample_bytree, 'colsample_bylevel':colsample_bylevel,
                       'colsample_bynode':colsample_bynode, 'reg_lambda':reg_lambda, 'num_parallel_trees':n_estimators}
        if eval_metric is not None:
            self.params['eval_metric'] = eval_metric
        if validate_parameters is not None:
            self.params['validate_parameters'] = validate_parameters
        if nthread is not None:
            self.params['nthread'] = nthread
        if disable_default_eval_metric is not None:
            self.params['disable_default_eval_metric'] = disable_default_eval_metric
        if gamma is not None:
            self.params['gamma'] = gamma
        if max_depth is not None:
            self.params['max_depth'] = max_depth
        if min_child_weight is not None:
            self.params['min_child_weight'] = min_child_weight
        if max_delta_step is not None:
            self.params['max_delta_step'] = max_delta_step
        if sampling_method is not None:
            self.params['sampling_method'] = sampling_method
        if reg_alpha is not None:
            self.params['reg_alpha'] = reg_alpha
        if tree_method is not None:
            self.params['tree_method'] = tree_method
        if grow_policy is not None:
            self.params['grow_policy'] = grow_policy
        if max_leaves is not None:
            self.params['max_leaves'] = max_leaves
        if random_state is not None:
            self.params['seed'] = random_state
        if seed_per_iteration is not None:
            self.params['seed_per_iteration'] = seed_per_iteration
        self.booster = None

    def train(self, dtrain, *, evals=None, obj=None, feval=None, maximize=None, 
              early_stopping_rounds=None, evals_result=None, verbose_eval=True, 
              rf_model=None, callbacks=None):
        
        # Allow training continuation from previous model
        if rf_model is not None:
            xgb_model = rf_model.booster
        else:
            xgb_model = None
        
        # Ensure num_boost_rounds is 1 for random forest
        nbr = 1
        training_args = {'num_boost_round':nbr}
        # Loop through optional arguments
        if evals is not None:
            training_args['evals'] = evals
        if obj is not None:
            training_args['obj'] = obj
        if feval is not None:
            training_args['feval'] = feval
        if maximize is not None:
            training_args['maximize'] = maximize
        if early_stopping_rounds is not None:
            training_args['early_stopping_rounds'] = early_stopping_rounds
        if evals_result is not None:
            training_args['evals_result'] = evals_result
        if verbose_eval is not None:
            training_args['verbose_eval'] = verbose_eval
        if xgb_model is not None:
            training_args['xgb_model'] = xgb_model
        if callbacks is not None:
            training_args['callbacks'] = callbacks

        self.booster = xgb.train(self.params, dtrain, **training_args)
        
    def predict(self, data, *, output_margin=False, ntree_limit=0, pred_leaf=False, 
                pred_contribs=False, approx_contribs=False, pred_interactions=False, 
                validate_features=True, training=False, iteration_range=(0, 0), 
                strict_shape=False):
        return self.booster.predict(data, output_margin=output_margin, 
                ntree_limit=ntree_limit, pred_leaf=pred_leaf, 
                pred_contribs=pred_contribs, approx_contribs=approx_contribs, 
                pred_interactions=pred_interactions, validate_features=validate_features, 
                training=training, iteration_range=iteration_range, strict_shape=strict_shape)
    
    def inplace_predict(self, data, *, iteration_range=(0, 0), predict_type='value', 
                missing=np.nan, validate_features=True, base_margin=None, strict_shape=False):
        return self.booster.inplace_predict(data, iteration_range=iteration_range, 
                predict_type=predict_type, missing=missing, validate_features=validate_features, 
                base_margin=base_margin, strict_shape=strict_shape)
        
    def save_model(self, fname):
        self.booster.save_model(fname)

    def load_model(self, fname):
        self.booster = xgb.Booster()
        self.booster.load_model(fname)    