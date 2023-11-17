// file that defines the corrections functions

#include <iostream>
#include <cmath>
#include "vec.h"
#include "mat.h"
#include "tools.h"

using namespace std;

// Declare base correction functions
// double* kendall_base(const vec diffs, const vec demographics, int& n);
// double* distance_base(const vec diffs, const vec demographics, int& n);

// setup ctypes externals declarations
extern "C" {
    // Define free ptr function
    void free_ptr(double* ptr){
        delete[] ptr;
    }

    // Declare Pearson Correction Functions
    double* pearson_raw_resid(const void* preds_inv, const void* true_inv, const void* demographics_inv, int n);
    double* pearson_abs_resid(const void* preds_inv, const void* true_inv, const void* demographics_inv, int n);
    double* pearson_square_resid(const void* preds_inv, const void* true_inv, const void* demographics_inv, int n);

    // Declare Distance Correction Functions
    double* distance_raw_resid(const void* preds_inv, const void* true_inv, const void* demographics_inv, int n);
    double* distance_abs_resid(const void* preds_inv, const void* true_inv, const void* demographics_inv, int n);
    double* distance_square_resid(const void* preds_inv, const void* true_inv, const void* demographics_inv, int n);

    // Declare Kendall Correction Functions
    double* kendall_raw_resid(const void* preds_inv, const void* true_inv, const void* demographics_inv, int n);
    double* kendall_abs_resid(const void* preds_inv, const void* true_inv, const void* demographics_inv, int n);
    double* kendall_square_resid(const void* preds_inv, const void* true_inv, const void* demographics_inv, int n);

    // Declare corrected loss functions
    double* pearson_corrected_loss(const void* preds_inv, const void* true_inv, const void* demographics_inv, int n, double gamma, int etype);
    double* distance_corrected_loss(const void* preds_inv, const void* true_inv, const void* demographics_inv, int n, double gamma, int etype);
    double* kendall_corrected_loss(const void* preds_inv, const void* true_inv, const void* demographics_inv, int n, double gamma, int etype);
}

// #######################################################################################################
// #######################################################################################################
// # 
// # Base MSE Definitions
// # 
// #######################################################################################################
// #######################################################################################################

double* mse(double* preds, double* true_vals, int n) {
    double* output = new double[2*n];
    for (int i = 0; i < n; i++) {
        output[i] = preds[i] - true_vals[i];
        output[i+n] = 1.0;
    }
    return output;
}

// #######################################################################################################
// #######################################################################################################
// # 
// # Pearson Correction Definitions
// # 
// #######################################################################################################
// #######################################################################################################

double* pearson_base(vec* diffs, vec* dems, int&n) {
    // Setup variables
    double mu_x = mean(diffs);
    double var_x = variance(diffs);
    double mu_d = mean(dems);
    double var_d = variance(dems);
    double cov = covariance(diffs, dems);
    vec dvar_x = *diffs - mu_x;
    dvar_x *= (2.*(1.-1./n));
    double d2var_x = 2.*pow((1.-1./n),2);
    vec dcov = *dems - mu_d;
    dcov *= (1.-1./n);
    vec dcov2 = dcov^2;
    vec dcovdvar = dcov * dvar_x;
    vec dvar2 = dvar_x^2;

    // Calculate the gradient
    vec grad = dcov * (2.*cov/var_x);
    grad -= dvar_x * pow(cov/var_x,2);
    grad /= var_d;

    // Calculate the hessian
    vec hess = dcov2 * (2./var_x);
    hess -= dcovdvar * (4.*cov/pow(var_x,2));
    hess -= d2var_x * pow(cov/var_x,2);
    hess += dvar2 * (2.*pow(cov,2)/pow(var_x,3));
    hess /= var_d;

    // Create output
    double* output = new double[2*n];
    // Add hess to the end of output
    for (int i = 0; i < n; i++){
        output[i] = grad.at(i);
        output[i+n] = hess.at(i);
    }
    return output;
}

double* pearson_raw_resid(const void* preds_inv, const void* true_inv, const void* demographics_inv, int n) {
    double * preds_in = (double*) preds_inv;
    double * true_in = (double*) true_inv;
    double * demographics_in = (double*) demographics_inv;

    vec preds(n, preds_in);
    vec true_vals(n, true_in);
    vec dems(n, demographics_in);
    vec diffs = preds - true_vals;

    double* output = pearson_base(&diffs, &dems, n);  
    return output;
}

double* pearson_abs_resid(const void* preds_inv, const void* true_inv, const void* demographics_inv, int n) {
    double * preds_in = (double*) preds_inv;
    double * true_in = (double*) true_inv;
    double * demographics_in = (double*) demographics_inv;

    vec preds(n, preds_in);
    vec true_vals(n, true_in);
    vec dems(n, demographics_in);
    vec diffs = preds - true_vals;
    vec signs = vec_sign(&diffs);
    diffs.set_abs();

    double* output = pearson_base(&diffs, &dems, n);  
    for (int i = 0; i < n; i++) {
        output[i] *= signs.at(i);
        output[i+n] *= signs.at(i);
    }
    return output;
}

double* pearson_square_resid(const void* preds_inv, const void* true_inv, const void* demographics_inv, int n) {
    double * preds_in = (double*) preds_inv;
    double * true_in = (double*) true_inv;
    double * demographics_in = (double*) demographics_inv;

    vec preds(n, preds_in);
    vec true_vals(n, true_in);
    vec dems(n, demographics_in);
    vec diffs = preds - true_vals;
    vec raw = diffs;
    diffs ^= 2;

    double* output = pearson_base(&diffs, &dems, n);  
    for (int i = 0; i < n; i++) {
        double grad = output[i] * 2. * raw.at(i);
        output[i] = grad;
        output[i+n] *= 2. * raw.at(i);
        output[i+n] += 2. * grad;
    }
    return output;
}

// #######################################################################################################
// #######################################################################################################
// # 
// # Distance Correction Definitions
// # 
// #######################################################################################################
// #######################################################################################################

double* distance_base(vec* diffs, vec* dems, int&n) {
    double* output = new double[2*n];

    // Setup variables
    mat A = difference_matrix(diffs, diffs);
    mat B = difference_matrix(dems, dems);
    mat S = matrix_sign(&A);
    vec S_row_sum = row_sum(&S);
    A.set_abs();
    center_difference_matrix(&A);
    B.set_abs();
    center_difference_matrix(&B);

    // Setup derivatives
    mat A2 = A^2;
    mat B2 = B^2;
    mat AB = A*B;
    double dVar_e = mean(&A2);
    double dVar_d = mean(&B2);
    double dCov = mean(&AB);

    // Setup loop variables
    mat del_Ajk(n);
    mat ind_matrix(n);
    vec rowi(n);
    vec coli(n);
    mat diff_mat(n);
    mat diff_ind(n);
    mat delAB(n);
    mat delAA(n);
    mat delA2(n);

    // Calculation loop
    for (int i = 0; i < n; i++) {
        // First calculate the del_Ajk matrix
        del_Ajk.set_all(2./pow(n,2));
        del_Ajk.add_to_row(i, -1./n);
        del_Ajk.add_to_col(i, -1./n);
        rowi = S.row(i);
        coli = S.col(i);
        del_Ajk *= sum(&rowi);
        // make ind_matrix where the i,j entry is 1 if i!= j and i!=k and 0 otherwise
        ind_matrix.set_all(1.);
        ind_matrix.set_row(i, 0.);
        ind_matrix.set_col(i, 0.);
        diff_mat = difference_matrix(&coli, &rowi);
        diff_ind = diff_mat*ind_matrix;
        diff_ind /= n;
        del_Ajk += diff_ind;
        // Calculate derivatives
        delAB = del_Ajk * B;
        delAA = del_Ajk * A;
        delA2 = del_Ajk^2;
        double del_dCov = mean(&delAB);
        double del_dVar_e = mean(&delAA);
        double del2_dVar_e = mean(&delA2);
        // Calculate the gradient
        output[i] = 2.* dCov * del_dCov / dVar_e;
        output[i] -= pow(dCov,2) * del_dVar_e / pow(dVar_e,2);
        output[i] /= dVar_d;
        // Calculate the hessian
        output[i+n] = 2.* pow(del_dCov,2) / dVar_e;
        output[i+n] -= 4. * dCov * del_dCov * del_dVar_e / pow(dVar_e,2);
        output[i+n] -= pow(dCov,2) * del2_dVar_e / pow(dVar_e,2);
        output[i+n] += 2. * pow(dCov * del_dVar_e,2) / pow(dVar_e,3);
        output[i+n] /= dVar_d;
    }
    
    return output;
}

double* distance_raw_resid(const void* preds_inv, const void* true_inv, const void* demographics_inv, int n) {
    double * preds_in = (double*) preds_inv;
    double * true_in = (double*) true_inv;
    double * demographics_in = (double*) demographics_inv;

    vec preds(n, preds_in);
    vec true_vals(n, true_in);
    vec demographics(n, demographics_in);
    vec diffs = preds - true_vals;

    double* output = distance_base(&diffs, &demographics, n);

    return output;    
}

double* distance_abs_resid(const void* preds_inv, const void* true_inv, const void* demographics_inv, int n) {
    double * preds_in = (double*) preds_inv;
    double * true_in = (double*) true_inv;
    double * demographics_in = (double*) demographics_inv;

    vec preds(n, preds_in);
    vec true_vals(n, true_in);
    vec dems(n, demographics_in);
    vec diffs = preds - true_vals;
    vec signs = vec_sign(&diffs);
    diffs.set_abs();

    double* output = distance_base(&diffs, &dems, n);
    for (int i = 0; i < n; i++) {
        output[i] *= signs.at(i);
        output[i+n] *= signs.at(i);
    }
    return output;    
}

double* distance_square_resid(const void* preds_inv, const void* true_inv, const void* demographics_inv, int n) {
    double * preds_in = (double*) preds_inv;
    double * true_in = (double*) true_inv;
    double * demographics_in = (double*) demographics_inv;

    vec preds(n, preds_in);
    vec true_vals(n, true_in);
    vec dems(n, demographics_in);
    vec diffs = preds - true_vals;
    vec raw = diffs;
    diffs ^= 2;

    double* output = distance_base(&diffs, &dems, n);  
    for (int i = 0; i < n; i++) {
        double grad = output[i] * 2. * raw.at(i);
        output[i] = grad;
        output[i+n] *= 2. * raw.at(i);
        output[i+n] += 2. * grad;
    }
    return output;  
}

// #######################################################################################################
// #######################################################################################################
// # 
// # Kendall Correction Definitions
// # 
// #######################################################################################################
// #######################################################################################################
double* kendall_base(vec* diffs, vec* dems, int&n) {
    double* output = new double[2*n];
    double grad_coeff = 2./(n*(n-1.));
    double hess_coeff = 4./(n*(n-1.));
    vec X = vec(*diffs);
    vec U = vec(*dems);
    sort_by(&X, &U); 

    double k_tau = kendalls_tau(&X, &U);

    for (int k=0; k<n; k++) {
        // Needed for all k
        double y_k = U.at(k);
        double x_k = X.at(k);
        double dtau_dk = 0.;
        double d2tau_dk2 = 0.;

        if (k == 0) {
            // Get k < n-1 values
            vec y_j = U.get_slice(k+1, n);
            vec x_j = X.get_slice(k+1, n);
            vec tempj = y_j - y_k; // Necessary to define outside of function call since argument has &
            vec sgn_1 = vec_sign(&tempj) * -1.;
            tempj = x_j - x_k;
            vec dk_1 = vec_cosh(&tempj); // Cosh is symmetric so *-1 not needed
            dk_1 ^= -2; 
            tempj = x_j - x_k;
            vec ddk_1 = veh_tanh(&tempj) * -1.; // Tanh is antisymmetric so *-1 outside

            vec temp1 = sgn_1 * dk_1;
            dtau_dk = grad_coeff * sum(&temp1) * k_tau;
            temp1 = sgn_1 * dk_1 * ddk_1;
            d2tau_dk2 = 2. * hess_coeff * (sum(&temp1) + dtau_dk);
        } else if (k == n-1) {
            // get k > 0 values
            vec y_i = U.get_slice(0, k);
            vec x_i = X.get_slice(0, k);                              
            vec tempi = y_i - y_k;            
            vec sgn_2 = vec_sign(&tempi) * -1.; 
            tempi = x_i - x_k;
            vec dk_2 = vec_cosh(&tempi);
            dk_2 ^= -2;
            tempi = x_i - x_k;
            vec ddk_2 = veh_tanh(&tempi);

            vec temp2 = sgn_2 * dk_2;
            dtau_dk = -grad_coeff * sum(&temp2) * k_tau;

            temp2 = sgn_2 * dk_2 * ddk_2;
            d2tau_dk2 = 2. * hess_coeff * (sum(&temp2) + dtau_dk);
        } else {
            // get k > 0 values
            vec y_i = U.get_slice(0, k);
            vec x_i = X.get_slice(0, k);                             
            vec tempi = y_i - y_k;            
            vec sgn_2 = vec_sign(&tempi) * -1.; 
            tempi = x_i - x_k;
            vec dk_2 = vec_cosh(&tempi);
            dk_2 ^= -2;
            tempi = x_i - x_k;
            vec ddk_2 = veh_tanh(&tempi);
            // get k < n-1 values
            vec y_j = U.get_slice(k+1, n);
            vec x_j = X.get_slice(k+1, n);
            vec tempj = y_j - y_k;
            vec sgn_1 = vec_sign(&tempj) * -1.;
            tempj = x_j - x_k;
            vec dk_1 = vec_cosh(&tempj); // Cosh is symmetric so *-1 not needed
            dk_1 ^= -2;
            tempj = x_j - x_k;
            vec ddk_1 = veh_tanh(&tempj) * -1.; // Tanh is antisymmetric so *-1 outside

            vec temp1 = sgn_1 * dk_1;
            vec temp2 = sgn_2 * dk_2;
            dtau_dk = k_tau * grad_coeff * (sum(&temp1) + sum(&temp2));

            temp1 = sgn_1 * dk_1 * ddk_1;
            temp2 = sgn_2 * dk_2 * ddk_2;
            d2tau_dk2 = 2. * hess_coeff * (sum(&temp1) + sum(&temp2) + dtau_dk);
        }
        output[k] = dtau_dk;
        output[k+n] = d2tau_dk2;
    }

    return output;
}

double* kendall_raw_resid(const void* preds_inv, const void* true_inv, const void* demographics_inv, int n) {
    double * preds_in = (double*) preds_inv;
    double * true_in = (double*) true_inv;
    double * demographics_in = (double*) demographics_inv;

    vec preds(n, preds_in);
    vec true_vals(n, true_in);
    vec demographics(n, demographics_in);
    vec diffs = preds - true_vals;

    double* output = kendall_base(&diffs, &demographics, n);

    return output;    
}

double* kendall_abs_resid(const void* preds_inv, const void* true_inv, const void* demographics_inv, int n) {
    double * preds_in = (double*) preds_inv;
    double * true_in = (double*) true_inv;
    double * demographics_in = (double*) demographics_inv;

    vec preds(n, preds_in);
    vec true_vals(n, true_in);
    vec dems(n, demographics_in);
    vec diffs = preds - true_vals;
    vec signs = vec_sign(&diffs);
    diffs.set_abs();

    double* output = kendall_base(&diffs, &dems, n);
    for (int i = 0; i < n; i++) {
        output[i] *= signs.at(i);
        output[i+n] *= signs.at(i);
    }
    return output;    
}

double* kendall_square_resid(const void* preds_inv, const void* true_inv, const void* demographics_inv, int n) {
    double * preds_in = (double*) preds_inv;
    double * true_in = (double*) true_inv;
    double * demographics_in = (double*) demographics_inv;

    vec preds(n, preds_in);
    vec true_vals(n, true_in);
    vec dems(n, demographics_in);
    vec diffs = preds - true_vals;
    vec raw = diffs;
    diffs ^= 2;

    double* output = kendall_base(&diffs, &dems, n);  
    for (int i = 0; i < n; i++) {
        double grad = output[i] * 2. * raw.at(i);
        output[i] = grad;
        output[i+n] *= 2. * raw.at(i);
        output[i+n] += 2. * grad;
    }
    return output;  
}


// #######################################################################################################
// #######################################################################################################
// # 
// # Combined Loss Definitions
// # 
// #######################################################################################################
// #######################################################################################################

double* pearson_corrected_loss(const void* preds_inv, const void* true_inv, const void* demographics_inv, int n, double gamma, int etype) {
    double* preds_in = (double*) preds_inv;
    double* true_in = (double*) true_inv;

    double* mse_loss = mse(preds_in, true_in, n);
    double* correction;
    if (etype==0){
        correction = pearson_raw_resid(preds_inv, true_inv, demographics_inv, n);
    } else if (etype==1) {
        correction = pearson_abs_resid(preds_inv, true_inv, demographics_inv, n);
    } else if (etype==2) {
        correction = pearson_square_resid(preds_inv, true_inv, demographics_inv, n);
    } 
    
    double* output = new double[2*n];
    for (int i = 0; i < n; i++) {
        output[i] = (1. - gamma) * mse_loss[i] + gamma * n * correction[i];
        output[i+n] = (1. - gamma) * mse_loss[i+n] + gamma * n * correction[i+n];
    }
    delete[] mse_loss;
    delete[] correction;
    return output;
}

double* distance_corrected_loss(const void* preds_inv, const void* true_inv, const void* demographics_inv, int n, double gamma, int etype) {
    double* preds_in = (double*) preds_inv;
    double* true_in = (double*) true_inv;

    double* mse_loss = mse(preds_in, true_in, n);
    double* correction;
    if (etype==0){
        correction = pearson_raw_resid(preds_inv, true_inv, demographics_inv, n);
    } else if (etype==1) {
        correction = pearson_abs_resid(preds_inv, true_inv, demographics_inv, n);
    } else if (etype==2) {
        correction = pearson_square_resid(preds_inv, true_inv, demographics_inv, n);
    } 
    double* output = new double[2*n];
    for (int i = 0; i < n; i++) {
        output[i] = (1. - gamma) * mse_loss[i] + gamma * n * correction[i];
        output[i+n] = (1. - gamma) * mse_loss[i+n] + gamma * n * correction[i+n];
    }
    delete[] mse_loss;
    delete[] correction;
    return output;
}

double* kendall_corrected_loss(const void* preds_inv, const void* true_inv, const void* demographics_inv, int n, double gamma, int etype) {
    double* preds_in = (double*) preds_inv;
    double* true_in = (double*) true_inv;

    double* mse_loss = mse(preds_in, true_in, n);
    double* correction;
    if (etype==0){
        correction = kendall_raw_resid(preds_inv, true_inv, demographics_inv, n);
    } else if (etype==1) {
        correction = kendall_abs_resid(preds_inv, true_inv, demographics_inv, n);
    } else if (etype==2) {
        correction = kendall_square_resid(preds_inv, true_inv, demographics_inv, n);
    } 
    double* output = new double[2*n];
    for (int i = 0; i < n; i++) {
        output[i] = (1. - gamma) * mse_loss[i] + gamma * n * correction[i];
        output[i+n] = (1. - gamma) * mse_loss[i+n] + gamma * n * correction[i+n];
    }
    delete[] mse_loss;
    delete[] correction;
    return output;
}

int main(void) {
    // Setup variables
    int n = 100;
    double* preds = new double[n];
    double* true_vals = new double[n];
    double* demographics = new double[n];
    for (int i = 0; i < n; i++) {
        preds[i] = 10*i;
        true_vals[i] = pow(i,2);
        demographics[i] = pow(i-1.,3);
    }

    // Run the function
    double* output = distance_raw_resid(preds, true_vals, demographics, n);

    // Print the output
    for (int i = 0; i < 2*n; i++) {
        std::cout << output[i] << std::endl;
    }
    delete[] output;
    delete[] preds;
    delete[] true_vals;
    delete[] demographics;
    return 0;
}