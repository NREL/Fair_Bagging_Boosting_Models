# Defines the function to test the bias of a model
import pandas as pd
import xgboost as xgb
from typing import Iterable
from statsmodels.stats.multitest import multipletests
import os
# add parent directory to path
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from tools.bias_utils import get_r2_pval
from xgb_wrappers.gradient_boosted_trees import GradientBoostedTreesModel
from xgb_wrappers.random_forest import RandomForestModel
from tools.loss_functions import get_pearson_corrected_mse, get_distance_corrected_mse, get_kendalls_corrected_mse

correction_dict = {'pearson':get_pearson_corrected_mse, 'distance':get_distance_corrected_mse, 'kendall':get_kendalls_corrected_mse}
model_dict = {'gbt':GradientBoostedTreesModel, 'rf':RandomForestModel}

def os_print(string):
    os.system('echo ' + repr(string))

def test_bias(model, test_data, dems, cutoffs=[0.6, 0.7, 0.8, 0.9], qvals=[0.05, 0.01, 0.001], return_bin_data=False):
    # Model should be an xgboost model, or the custom xgb random forest / gradient boosted trees model
    # Test data should be a DMatrix with final column the demographics
    # Dem is a dataframe containing the demographic information

    if not isinstance(test_data, xgb.DMatrix):
        raise TypeError('test_data must be an xgb.DMatrix')
    if not isinstance(cutoffs, Iterable):
        cutoffs = [cutoffs]
    if not isinstance(qvals, Iterable):
        qvals = [qvals]

    # Get the predictions
    ypred = model.predict(test_data)
    # Get the actual values
    ytrue = test_data.get_label()
    # Make dataframe
    bias_data = pd.DataFrame({'Volume':ytrue, 'PredVolume':ypred})
    bias_data = bias_data.reset_index(drop=True)
    if not (isinstance(dems, pd.DataFrame) or isinstance(dems, pd.Series)):
        dems = pd.DataFrame({'Demographic':dems.flatten()})
    elif isinstance(dems, pd.Series):
        dems = pd.DataFrame({dems.name:dems})
    dems = dems.reset_index(drop=True)
    bias_data = pd.concat([bias_data, dems], axis=1)
    # return bias_data

    # Create binary cutoff demographic columns
    cols = []
    for dem in dems.columns:
        for cutoff in cutoffs:
            col = f'{dem}_{cutoff}'
            cols.append(col)
            bias_data[col] = bias_data[dem].apply(lambda x: 1 if x > cutoff else 0)

    # Get the r2 and pvalues
    r2_0_vals = []
    r2_1_vals = []
    pvals = []
    # print(bias_data)
    for col in cols:
        r2_0, r2_1, pval = get_r2_pval(bias_data, col)
        r2_1_vals.append(r2_1)
        r2_0_vals.append(r2_0)
        pvals.append(pval)
    test_data = pd.DataFrame({'Demographic': cols, 'r2_0': r2_0_vals, 'r2_1': r2_1_vals, 'pval': pvals})

    # apply the fdr correction
    test_data['fdr_corrected_pval'] = multipletests(test_data['pval'], method='fdr_bh')[1]
    test_data['r2_diff'] = abs(test_data['r2_1'] - test_data['r2_0'])
    for qval in qvals:
        test_data[f'Sig at {qval}'] = test_data['fdr_corrected_pval'] < qval

    if return_bin_data:
        return test_data, bias_data
    else:
        return test_data
    
