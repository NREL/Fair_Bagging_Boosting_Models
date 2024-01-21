import numpy as np 
import xgboost as xgb
from typing import Tuple, Callable
import pandas as pd
from dmatrix2np import dmatrix_to_numpy
import os 
from ctypes import cdll, c_double, c_int, POINTER
from numpy.ctypeslib import ndpointer, as_ctypes
import pandas as pd
from scipy import stats
from datetime import datetime
import sys 
import os
current_dir = os.path.dirname(os.path.realpath(__file__))

#######################################################################################################
#######################################################################################################
# 
# Corrected Loss Functions
# 
#######################################################################################################
#######################################################################################################
def get_pearson_corrected_mse(gamma: float, etype: int = 0, compiled: bool=True) -> Callable[[np.ndarray, xgb.DMatrix], Tuple[np.ndarray, np.ndarray]]:
    if not os.path.exists(f'{current_dir}/compiled_loss_funcs/libcorrections.so'):
        os.system(f'rm {current_dir}/compiled_loss_funcs/libcorrections.so ./tools/compiled_loss_funcs/corrections.o')
        os.system(f'g++ -c -fPIC {current_dir}/compiled_loss_funcs/corrections.cpp -o ./tools/compiled_loss_funcs/corrections.o')
        os.system(f'g++ -shared -o {current_dir}/compiled_loss_funcs/libcorrections.so ./tools/compiled_loss_funcs/corrections.o')

    lib = cdll.LoadLibrary(f'{current_dir}/compiled_loss_funcs/libcorrections.so')
        
    # We use the correction to define an internal loss function
    ff = lib.pearson_corrected_loss
    ff.argtypes = [ndpointer(c_double, flags="C_CONTIGUOUS"),  # Tell python the data types of the inputs
                    ndpointer(c_double, flags="C_CONTIGUOUS"),
                    ndpointer(c_double, flags="C_CONTIGUOUS"), 
                    c_int, c_double, c_int]
    ff.restype = POINTER(c_double) # Tell python the output is a pointer to a double
    def mse_pearson_corrected(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        n = len(predt)
        dems = dmatrix_to_numpy(dtrain)[:, -1].astype(np.float64)
        true_vals = dtrain.get_label().astype(np.float64)
        pred = predt.astype(np.float64)

        outptr = ff(pred, true_vals, dems, n, gamma, etype)
        c_out = np.ctypeslib.as_array(outptr, shape=(2*n,)) # Convert to numpy array
        grad = c_out[:n]
        hess = c_out[n:]
        lib.free_ptr(outptr)
        return grad, hess
    # Get pearson corrected mse returns the callable mse_pearson_corrected function
    return mse_pearson_corrected

def get_distance_corrected_mse(gamma: float, etype: int = 0, compiled: bool=True) -> Callable[[np.ndarray, xgb.DMatrix], Tuple[np.ndarray, np.ndarray]]:
    if not os.path.exists(f'{current_dir}/compiled_loss_funcs/libcorrections.so'):
        os.system(f'rm {current_dir}/compiled_loss_funcs/libcorrections.so ./tools/compiled_loss_funcs/corrections.o')
        os.system(f'g++ -c -fPIC {current_dir}/compiled_loss_funcs/corrections.cpp -o ./tools/compiled_loss_funcs/corrections.o')
        os.system(f'g++ -shared -o {current_dir}/compiled_loss_funcs/libcorrections.so ./tools/compiled_loss_funcs/corrections.o')
    
    lib = cdll.LoadLibrary(f'{current_dir}/compiled_loss_funcs/libcorrections.so')

    ff = lib.distance_corrected_loss
    ff.argtypes = [ndpointer(c_double, flags="C_CONTIGUOUS"),  # Tell python the data types of the inputs
                    ndpointer(c_double, flags="C_CONTIGUOUS"),
                    ndpointer(c_double, flags="C_CONTIGUOUS"), 
                    c_int, c_double, c_int]
    ff.restype = POINTER(c_double) # Tell python the output is a pointer to a double
    def mse_distance_corrected(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        n = len(predt)
        dems = dmatrix_to_numpy(dtrain)[:, -1].astype(np.float64)
        true_vals = dtrain.get_label().astype(np.float64)
        pred = predt.astype(np.float64)

        outptr = ff(pred, true_vals, dems, n, gamma, etype)
        c_out = np.ctypeslib.as_array(outptr, shape=(2*n,)) # Convert to numpy array
        grad = c_out[:n]
        hess = c_out[n:]
        lib.free_ptr(outptr)
        return grad, hess
    return mse_distance_corrected

def get_kendalls_corrected_mse(gamma: float, etype: int = 0, compiled: bool=True) -> Callable[[np.ndarray, xgb.DMatrix], Tuple[np.ndarray, np.ndarray]]:
    if not os.path.exists(f'{current_dir}/compiled_loss_funcs/libcorrections.so'):
        os.system(f'rm {current_dir}/compiled_loss_funcs/libcorrections.so ./tools/compiled_loss_funcs/corrections.o')
        os.system(f'g++ -c -fPIC {current_dir}/compiled_loss_funcs/corrections.cpp -o ./tools/compiled_loss_funcs/corrections.o')
        os.system(f'g++ -shared -o {current_dir}/compiled_loss_funcs/libcorrections.so ./tools/compiled_loss_funcs/corrections.o')
    
    lib = cdll.LoadLibrary(f'{current_dir}/compiled_loss_funcs/libcorrections.so')

    ff = lib.kendall_corrected_loss
    ff.argtypes = [ndpointer(c_double, flags="C_CONTIGUOUS"),  # Tell python the data types of the inputs
                    ndpointer(c_double, flags="C_CONTIGUOUS"),
                    ndpointer(c_double, flags="C_CONTIGUOUS"), 
                    c_int, c_double, c_int]
    ff.restype = POINTER(c_double) # Tell python the output is a pointer to a double
    def mse_kendall_corrected(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        n = len(predt)
        dems = dmatrix_to_numpy(dtrain)[:, -1].astype(np.float64)
        true_vals = dtrain.get_label().astype(np.float64)
        pred = predt.astype(np.float64)

        outptr = ff(pred, true_vals, dems, n, gamma, etype)
        c_out = np.ctypeslib.as_array(outptr, shape=(2*n,)) # Convert to numpy array
        grad = c_out[:n]
        hess = c_out[n:]
        lib.free_ptr(outptr)
        return grad, hess
    return mse_kendall_corrected

