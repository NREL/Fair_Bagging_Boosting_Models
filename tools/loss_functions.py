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

fully_compiled = True

#######################################################################################################
#######################################################################################################
# 
# Base Loss Functions 
# 
#######################################################################################################
#######################################################################################################
# MSE Loss
def mse_loss(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        # l(y_val, y_pred) = 1/2 (y_val-y_pred)**2
        y_val = dtrain.get_label() # Get label returns the true y values
        return (predt-y_val) # Predt is the predicted y values
    def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        # l(y_val, y_pred) = 1/2 (y_val-y_pred)**2
        return np.ones_like(predt)
    grad = gradient(predt, dtrain)
    hess = hessian(predt, dtrain)
    return grad, hess

#######################################################################################################
#######################################################################################################
# 
# Correlation Penalties 
# 
#######################################################################################################
#######################################################################################################
# Pearson Correlation Penalty
def pearson_correlation_penalty_squared_resid(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    # r = Cov(X,Y) / (std(X) * std(Y))
    # r^2 = Cov(X,Y)^2 / (Var(X) * Var(Y))
    # Cov(X,Y) = sum((x_i - mean(X)) * (y_i - mean(Y))) / (n-1)
    # Var(X) = sum((x_i - mean(X))**2) / (n-1)
    diffs = predt - dtrain.get_label() #predt is the predicted y values
                                        # .get_label() returns the true y values
    rawvals = diffs # Raw residual
    diffs = diffs**2 # Squared residual
    dems = dmatrix_to_numpy(dtrain)[:, -1] # Here is where we are getting the demographics
                                            # dmatrix_to_numpy returns a numpy array of the dmatrix
                                            # The last column is the demographics
    n = len(diffs)
    mu_x = np.mean(diffs)
    var_x = np.sum((diffs-mu_x)**2) # Denominator cancels out
    dvar_x = 2*(1.-1./n)*(diffs - mu_x)
    d2var_x = 2*(1. - 1./n)**2 # d2cov = 0
    mu_d = np.mean(dems)
    cov = np.sum((diffs-mu_x)*(dems-mu_d)) # Denominator cancels out
    dcov = (1. - 1./n)*(dems - mu_d)
    var_d = np.sum((dems-mu_d)**2) # Denominator cancels out

    # Calculate the Gradient
    grad = 2*cov*dcov / var_x 
    grad -= cov**2 * dvar_x / var_x**2 
    grad /= var_d 
    grad *= 2*rawvals # de/dy = 2(yhat - y)

    # Calculate the Hessian
    hess = 2*dcov**2 / var_x
    hess -= 4*cov*dcov*dvar_x / var_x**2
    hess -= cov**2 * d2var_x / var_x**2 
    hess += 2*cov**2 * dvar_x**2 / var_x**3
    hess /= var_d
    hess *= 2*rawvals # de/dy = 2(yhat - y)
    hess += 2*grad # This is the dr/de * d2e/dy2 term
    return grad, hess

def pearson_correlation_penalty_abs_resid(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    # r = Cov(X,Y) / (std(X) * std(Y))
    # r^2 = Cov(X,Y)^2 / (Var(X) * Var(Y))
    # Cov(X,Y) = sum((x_i - mean(X)) * (y_i - mean(Y))) / (n-1)
    # Var(X) = sum((x_i - mean(X))**2) / (n-1)
    diffs = predt - dtrain.get_label()
    signs = np.sign(diffs) 
    diffs = np.abs(diffs)
    dems = dmatrix_to_numpy(dtrain)[:, -1]
    n = len(diffs)
    mu_x = np.mean(diffs)
    var_x = np.sum((diffs-mu_x)**2) # Denominator cancels out
    dvar_x = 2*(1.-1./n)*(diffs - mu_x)
    d2var_x = 2*(1. - 1./n)**2 # d2cov = 0
    mu_d = np.mean(dems)
    cov = np.sum((diffs-mu_x)*(dems-mu_d)) # Denominator cancels out
    dcov = (1. - 1./n)*(dems - mu_d)
    var_d = np.sum((dems-mu_d)**2) # Denominator cancels out

    # Calculate the Gradient
    grad = 2*cov*dcov / var_x 
    grad -= cov**2 * dvar_x / var_x**2 
    grad /= var_d 
    grad *= signs # dr^2/de * de/dy

    # Calculate the Hessian
    hess = 2*dcov**2 / var_x
    hess -= 4*cov*dcov*dvar_x / var_x**2
    hess -= cov**2 * d2var_x / var_x**2 
    hess += 2*cov**2 * dvar_x**2 / var_x**3
    hess /= var_d
    hess *= signs # d/dy dr/de = d2r/de2 * de/dy + dr/de * d2e/dy2
    return grad, hess

def pearson_correlation_penalty_raw_resid(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    # r = Cov(X,Y) / (std(X) * std(Y))
    # r^2 = Cov(X,Y)^2 / (Var(X) * Var(Y))
    # Cov(X,Y) = sum((x_i - mean(X)) * (y_i - mean(Y))) / (n-1)
    # Var(X) = sum((x_i - mean(X))**2) / (n-1)
    diffs = predt - dtrain.get_label()
    dems = dmatrix_to_numpy(dtrain)[:, -1]
    n = len(diffs)
    mu_x = np.mean(diffs)
    var_x = np.sum((diffs-mu_x)**2) # Denominator cancels out
    dvar_x = 2*(1.-1./n)*(diffs - mu_x)
    d2var_x = 2*(1. - 1./n)**2 # d2cov = 0
    mu_d = np.mean(dems)
    cov = np.sum((diffs-mu_x)*(dems-mu_d)) # Denominator cancels out
    dcov = (1. - 1./n)*(dems - mu_d)
    var_d = np.sum((dems-mu_d)**2) # Denominator cancels out

    # Calculate the Gradient
    grad = 2*cov*dcov / var_x 
    grad -= cov**2 * dvar_x / var_x**2 
    grad /= var_d 

    # Calculate the Hessian
    hess = 2*dcov**2 / var_x
    hess -= 4*cov*dcov*dvar_x / var_x**2
    hess -= cov**2 * d2var_x / var_x**2 
    hess += 2*cov**2 * dvar_x**2 / var_x**3
    hess /= var_d
    return grad, hess

def frobenius_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a*b)

def distance_correlation_penalty(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    e = (predt - dtrain.get_label()).astype(np.float64)
    d = dmatrix_to_numpy(dtrain)[:, -1].astype(np.float64)
    n = len(e)
    # difference matrices
    A = e[:, None] - e[None, :]
    B = d[:, None] - d[None, :]
    # take the sign of the difference matrices
    S = np.sign(A)
    # double centered difference matrices
    A = np.abs(A)
    B = np.abs(B)
    A_row_mean = np.mean(A, axis=1)
    A_col_mean = np.mean(A, axis=0)
    B_row_mean = np.mean(B, axis=1)
    B_col_mean = np.mean(B, axis=0)
    A_mean = np.mean(A)
    B_mean = np.mean(B)
    # center the difference matrices
    A -= A_row_mean[:, None]
    A -= A_col_mean[None, :]
    A += A_mean
    B -= B_row_mean[:, None]
    B -= B_col_mean[None, :]
    B += B_mean
    A_row_mean = np.mean(A, axis=1)
    # print("python\n", A)
    dVar_e = np.sum(A**2) / (n**2)
    dVar_d = np.sum(B**2) / (n**2)
    dCov = np.sum(A*B) / (n**2)
    hess = np.zeros_like(e)
    grad = np.zeros_like(e)
    for i in range(n):
        # First calculate the del_Ajk matrix
        del_Ajk = np.ones((n, n)) * 2/n**2
        del_Ajk[i,:] -= 1/n 
        del_Ajk[:,i] -= 1/n
        del_Ajk *= np.sum(S[i,:])
        # make ind_matrix where the i,j entry is 1 if i!= j and i!=k and 0 otherwise
        ind_matrix = np.ones((n, n))
        ind_matrix[i,:] = 0
        ind_matrix[:,i] = 0
        del_Ajk += (S[:,i][:,None] - S[i,:][None,:]) * ind_matrix / n
        # calculate derivatives 
        del_dCov = np.sum(del_Ajk * B) / (n**2)
        del_dVar_e = np.sum(del_Ajk * A) / (n**2)
        del2_dvar_e = np.sum(del_Ajk**2) / (n**2)
        # Calculate gradient
        grad[i] = 2*dCov*del_dCov / dVar_e
        grad[i] -= dCov**2 * del_dVar_e / dVar_e**2
        grad[i] /= dVar_d
        # Calculate Hessian
        hess[i] = 2*del_dCov**2 / dVar_e
        hess[i] -= 4*dCov*del_dCov*del_dVar_e / dVar_e**2
        hess[i] -= dCov**2 * del2_dvar_e / dVar_e**2
        hess[i] += 2*dCov**2 * del_dVar_e**2 / dVar_e**3
        hess[i] /= dVar_d
    return grad, hess


def kendalls_tau(x, y):
    n = len(x)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i+1, n):
            if (x[i] - x[j]) * (y[i] - y[j]) > 0:
                concordant += 1
            elif (x[i] - x[j]) * (y[i] - y[j]) < 0:
                discordant += 1
    return (concordant - discordant) / (n*(n-1)/2)

def kendalls_correlation_penalty_raw_resid(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    
    diffs = predt - dtrain.get_label()
    dems = dmatrix_to_numpy(dtrain)[:, -1]
    tmp = pd.DataFrame()
    tmp['diffs'] = diffs
    tmp['dems'] = dems
    tmp_sorted = tmp.sort_values(by='diffs')
    n = len(diffs)
    grad_coeff = 2/(n*(n-1))
    hess_coeff = 4/(n*(n-1))
    
    
    X = list(tmp_sorted['diffs'])
    U = list(tmp_sorted['dems'])
   
    grad = []
    hess = []
   
    k_tau = kendalls_tau(diffs, dems)
    now = datetime.now()
    print("starts here",now.strftime("%H:%M:%S"))
    for k in np.arange(0,n,1):
        
        y_k = U[k]
        y_j = np.array(U[(k+1):-1])
        y_i = np.array(U[0:k])
        
        x_i = np.array(X[0:k])
        x_k = X[k]
        x_j = np.array(X[(k+1):-1])
        
        sgn_1 = np.sign(y_k - y_j)
        dk_1 = np.cosh(x_k - x_j)**(-2)
        
        sgn_2 = -np.sign(y_i - y_k)
        dk_2 = np.cosh(x_i - x_k)**(-2)
        
        ddk_1 = np.tanh(x_k - x_j)
        ddk_2 = np.tanh(x_i - x_k)

        
        
        if k == 0:
            
            dtau_dk = grad_coeff*np.sum(sgn_1*dk_1) * k_tau   
            d2tau_dk2 = 2 * hess_coeff* (np.sum(sgn_1*dk_1*ddk_1) + dtau_dk)
        elif k == (n-1):
           
            dtau_dk = -grad_coeff*np.sum(sgn_2*dk_2) * k_tau        
            d2tau_dk2 = 2 * hess_coeff* (np.sum(sgn_2*dk_2*ddk_2) + dtau_dk)
        else: 
            
            dtau_dk = k_tau * grad_coeff*(np.sum(sgn_1*dk_1) + np.sum(sgn_2*dk_2))
            d2tau_dk2 = 2 * hess_coeff*(np.sum(sgn_1*dk_1*ddk_1) + np.sum(sgn_2*dk_2*ddk_2) + dtau_dk)
        
        grad.append(dtau_dk)
        hess.append(d2tau_dk2)
    return np.array(grad), np.array(hess)

#######################################################################################################
#######################################################################################################
# 
# Corrected Loss Functions
# 
#######################################################################################################
#######################################################################################################
pearson_etypes = {0: pearson_correlation_penalty_raw_resid, 1:pearson_correlation_penalty_abs_resid, 2:pearson_correlation_penalty_squared_resid}
def get_pearson_corrected_mse(gamma: float, etype: int = 0, compiled: bool=True) -> Callable[[np.ndarray, xgb.DMatrix], Tuple[np.ndarray, np.ndarray]]:
    if compiled: 
        if not os.path.exists(f'{current_dir}/compiled_loss_funcs/libcorrections.so'):
            os.system(f'rm {current_dir}/compiled_loss_funcs/libcorrections.so ./tools/compiled_loss_funcs/corrections.o')
            os.system(f'g++ -c -fPIC {current_dir}/compiled_loss_funcs/corrections.cpp -o ./tools/compiled_loss_funcs/corrections.o')
            os.system(f'g++ -shared -o {current_dir}/compiled_loss_funcs/libcorrections.so ./tools/compiled_loss_funcs/corrections.o')

        lib = cdll.LoadLibrary(f'{current_dir}/compiled_loss_funcs/libcorrections.so')
        if etype == 0:
            f = lib.pearson_raw_resid
        elif etype == 1:
            f = lib.pearson_abs_resid
        elif etype == 2:
            f = lib.pearson_squared_resid
        else:
            raise ValueError('Compiled pearson loss function only supports etype 0, 1 and 2')
        
        f.argtypes = [ndpointer(c_double, flags="C_CONTIGUOUS"),  # Tell python the data types of the inputs
                        ndpointer(c_double, flags="C_CONTIGUOUS"),
                        ndpointer(c_double, flags="C_CONTIGUOUS"), 
                        c_int]
        f.restype = POINTER(c_double) # Tell python the output is a pointer to a double
        
        def correction(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
            n = len(predt)
            dems = dmatrix_to_numpy(dtrain)[:, -1].astype(np.float64) # Inputs must be np.float64!!!
            true_vals = dtrain.get_label().astype(np.float64)
            pred = predt.astype(np.float64)

            outptr = f(pred, true_vals, dems, n) # Returns a pointer
            c_out = np.ctypeslib.as_array(outptr, shape=(2*n,)) # Convert to numpy array
            lib.free_ptr(outptr)
            grad = c_out[:n]
            hess = c_out[n:]
            return grad, hess
        
    else:
        correction = pearson_etypes[etype] # Get correction function
                                            # 0 is raw residuals
                                            # 1 is absolute residuals
                                            # 2 is squared residuals
    # We use the correction to define an internal loss function
    use_python = not (compiled and fully_compiled)
    if use_python:
        def mse_pearson_corrected(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
            grad_mse, hess_mse = mse_loss(predt, dtrain) 
            grad_corr, hess_corr = correction(predt, dtrain)
            n = len(grad_mse)
            grad = (1-gamma)*grad_mse + n*gamma*grad_corr
            hess = (1-gamma)*hess_mse + n*gamma*hess_corr
            return grad, hess
    else:
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
    if compiled: 
        if not os.path.exists(f'{current_dir}/compiled_loss_funcs/libcorrections.so'):
            os.system(f'rm {current_dir}/compiled_loss_funcs/libcorrections.so ./tools/compiled_loss_funcs/corrections.o')
            os.system(f'g++ -c -fPIC {current_dir}/compiled_loss_funcs/corrections.cpp -o ./tools/compiled_loss_funcs/corrections.o')
            os.system(f'g++ -shared -o {current_dir}/compiled_loss_funcs/libcorrections.so ./tools/compiled_loss_funcs/corrections.o')
        
        lib = cdll.LoadLibrary(f'{current_dir}/compiled_loss_funcs/libcorrections.so')
        if etype == 0:
            f = lib.distance_raw_resid
        elif etype == 1:    
            f = lib.distance_abs_resid
        elif etype == 2:
            f = lib.distance_square_resid
        else:
            raise ValueError('Compiled distance correlation loss function only supports etype 0, 1 and 2')
        
        f.argtypes = [ndpointer(c_double, flags="C_CONTIGUOUS"),  # Tell python the data types of the inputs
                        ndpointer(c_double, flags="C_CONTIGUOUS"),
                        ndpointer(c_double, flags="C_CONTIGUOUS"),
                        c_int]
        f.restype = POINTER(c_double) # Tell python the output is a pointer to a double
        
        def correction(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
            n = len(predt)
            dems = dmatrix_to_numpy(dtrain)[:, -1].astype(np.float64) # Inputs must be np.float64!!!
            true_vals = dtrain.get_label().astype(np.float64)
            pred = predt.astype(np.float64)

            outptr = f(pred, true_vals, dems, n) # Returns a pointer
            c_out = np.ctypeslib.as_array(outptr, shape=(2*n,)) # Convert to numpy array
            grad = c_out[:n]
            hess = c_out[n:]
            lib.free_ptr(outptr)
            return grad, hess
    else:
        correction = distance_correlation_penalty
    use_python = not (compiled and fully_compiled)
    if use_python:
        def mse_distance_corrected(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
            grad_mse, hess_mse = mse_loss(predt, dtrain) 
            grad_corr, hess_corr = correction(predt, dtrain)
            n = len(grad_mse)
            grad = (1-gamma)*grad_mse + n*gamma*grad_corr
            hess = (1-gamma)*hess_mse + n*gamma*hess_corr
            return grad, hess
    else:
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

kendalls_etypes = {0: kendalls_correlation_penalty_raw_resid} #, 1: kendalls_correlation_penalty_abs_resid, 2: kendalls_correlation_penalty_squared_resid}
def get_kendalls_corrected_mse(gamma: float, etype: int = 0, compiled: bool=True) -> Callable[[np.ndarray, xgb.DMatrix], Tuple[np.ndarray, np.ndarray]]:
    if compiled: 
        if not os.path.exists(f'{current_dir}/compiled_loss_funcs/libcorrections.so'):
            os.system(f'rm {current_dir}/compiled_loss_funcs/libcorrections.so ./tools/compiled_loss_funcs/corrections.o')
            os.system(f'g++ -c -fPIC {current_dir}/compiled_loss_funcs/corrections.cpp -o ./tools/compiled_loss_funcs/corrections.o')
            os.system(f'g++ -shared -o {current_dir}/compiled_loss_funcs/libcorrections.so ./tools/compiled_loss_funcs/corrections.o')
        
        lib = cdll.LoadLibrary(f'{current_dir}/compiled_loss_funcs/libcorrections.so')
        if etype == 0:
            f = lib.kendall_raw_resid
        elif etype == 1:    
            f = lib.kendall_abs_resid
        elif etype == 2:
            f = lib.kendall_square_resid
        else:
            raise ValueError('Compiled distance correlation loss function only supports etype 0, 1 and 2')
        
        f.argtypes = [ndpointer(c_double, flags="C_CONTIGUOUS"),  # Tell python the data types of the inputs
                        ndpointer(c_double, flags="C_CONTIGUOUS"),
                        ndpointer(c_double, flags="C_CONTIGUOUS"),
                        c_int]
        f.restype = POINTER(c_double) # Tell python the output is a pointer to a double
        
        def correction(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
            n = len(predt)
            dems = dmatrix_to_numpy(dtrain)[:, -1].astype(np.float64) # Inputs must be np.float64!!!
            true_vals = dtrain.get_label().astype(np.float64)
            pred = predt.astype(np.float64)

            outptr = f(pred, true_vals, dems, n) # Returns a pointer
            c_out = np.ctypeslib.as_array(outptr, shape=(2*n,)) # Convert to numpy array
            grad = c_out[:n]
            hess = c_out[n:]
            lib.free_ptr(outptr)
            return grad, hess
    else:
        correction = kendalls_etypes[etype] # Get correction function
                                            # 0 is raw residuals
                                            # 1 is absolute residuals
                                            # 2 is squared residuals
    use_python = not (compiled and fully_compiled)
    if use_python:
        def mse_kendall_corrected(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
            grad_mse, hess_mse = mse_loss(predt, dtrain) 
            grad_corr, hess_corr = correction(predt, dtrain)
            n = len(grad_mse)
            grad = (1-gamma)*grad_mse + n*gamma*grad_corr
            hess = (1-gamma)*hess_mse + n*gamma*hess_corr
            return grad, hess
    else:
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

