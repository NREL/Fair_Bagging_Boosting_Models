#######################################################################################################
#######################################################################################################
# Gradient Boosted Trees Model defined within XGBoost framework
# Created by Erik Bensen, National Renewable Energy Laboratory
# 5/12/2023
#######################################################################################################
#######################################################################################################

import xgboost as xgb
import numpy as np 
import pandas as pd

class GradientBoostedTreesModel():
    def __init__(self, objective='reg:squarederror', verbosity=1, subsample=0.8, colsample_bytree=1.0, eta=None,
                 num_boost_round=100, *, eval_metric=None, validate_parameters=None, 
                 nthread=None, disable_default_eval_metric=None,
                 max_depth=None, min_child_weight=None, max_delta_step=None,
                 tree_method=None, random_state=None, seed_per_iteration=None):
        # From what I've found, the primary differences between XGB and GBT are:
        # 1. GBT has no L1 or L2 regularization
        # 2. GBT only has colsample_bytree, not colsample_bylevel or colsample_bynode
        # 3. GBT has no sampling_method, fixes to uniform
        # 4. GBT has no grow_policy, fixes to depthwise
        # 5. GBT has no max_leaves, fixes to 0 -- uses max_depth instead
        # 6. GBT has no pruning 
        self.nbr = num_boost_round
        self.params = {'objective':objective, 'booster':'gbtree', 'verbosity':verbosity,
                       'subsample':subsample, 'colsample_bytree':colsample_bytree}
        if eta is not None:
            self.params['eta'] = eta
        if eval_metric is not None:
            self.params['eval_metric'] = eval_metric
        if validate_parameters is not None:
            self.params['validate_parameters'] = validate_parameters
        if nthread is not None:
            self.params['nthread'] = nthread
        if disable_default_eval_metric is not None:
            self.params['disable_default_eval_metric'] = disable_default_eval_metric
        if max_depth is not None:
            self.params['max_depth'] = max_depth
        if min_child_weight is not None:
            self.params['min_child_weight'] = min_child_weight
        if max_delta_step is not None:
            self.params['max_delta_step'] = max_delta_step
        if tree_method is not None:
            self.params['tree_method'] = tree_method
        if random_state is not None:
            self.params['random_state'] = random_state
        if seed_per_iteration is not None:
            self.params['seed_per_iteration'] = seed_per_iteration
        

    def train(self, dtrain, *, evals=None, obj=None, feval=None, maximize=None, 
              early_stopping_rounds=None, evals_result=None, verbose_eval=True, 
              gbt_model=None, callbacks=None):

        # Allow training continuation from previous model
        if gbt_model is not None:
            xgb_model = xgb.Booster()
        else:
            xgb_model = None

        training_args = {'num_boost_round':self.nbr}
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
        if gbt_model is not None:
            training_args['xgb_model'] = xgb_model
        if callbacks is not None:
            training_args['callbacks'] = callbacks

        self.booster = xgb.train(self.params, dtrain, **training_args)
        
        pass
        
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