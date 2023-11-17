#include <cmath>
#include "vec.h"
#include "mat.h"
#include <algorithm>

#pragma once

using namespace std;

double mean(vec* x) {
    double sum = 0.0;
    for (int i = 0; i < x->get_n(); i++) {
        sum += x->at(i);
    }
    return sum / x->get_n();
}
double mean(mat* x) {
    double sum = 0.0;
    for (int i = 0; i < x->get_n(); i++) {
        for (int j = 0; j < x->get_n(); j++) {
            sum += x->at(i, j);
        }
    }
    return sum / (x->get_n() * x->get_n());
}
double sum(vec* x) {
    double sum = 0.0;
    for (int i = 0; i < x->get_n(); i++) {
        sum += x->at(i);
    }
    return sum;
}
double sum(mat* x) {
    double sum = 0.0;
    for (int i = 0; i < x->get_n(); i++) {
        for (int j = 0; j < x->get_n(); j++) {
            sum += x->at(i, j);
        }
    }
    return sum;
}

vec row_sum(mat* x) {
    int n = x->get_n();
    vec result(n);
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += x->at(i, j);
        }
        result.set(i, sum);
    }
    return result;
}
vec row_mean(mat* x) {
    int n = x->get_n();
    vec result(n);
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += x->at(i, j);
        }
        result.set(i, sum / n);
    }
    return result;
}
vec col_mean(mat* x) {
    int n = x->get_n();
    vec result(n);
    for (int j = 0; j < n; j++) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += x->at(i, j);
        }
        result.set(j, sum / n);
    }
    return result;
}
vec col_sum(mat* x) {
    int n = x->get_n();
    vec result(n);
    for (int j = 0; j < n; j++) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += x->at(i, j);
        }
        result.set(j, sum);
    }
    return result;
}

double variance(vec* x) {
    double x_mean = mean(x);
    double sum = 0.0;
    for (int i = 0; i < x->get_n(); i++) {
        sum += pow(x->at(i) - x_mean, 2);
    }
    return sum; // Denomenator cancels out
}
double covariance(vec* x, vec* y) {
    double x_mean = mean(x);
    double y_mean = mean(y);
    double sum = 0.0;
    for (int i = 0; i < x->get_n(); i++) {
        sum += (x->at(i) - x_mean) * (y->at(i) - y_mean);
    }
    return sum; // Denomenator cancels out
}

mat difference_matrix(vec* x, vec* y) {
    int n = x->get_n();
    mat result(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result.set(i, j, x->at(i) - y->at(j));
        }
    }
    return result;
}

void center_difference_matrix(mat* v) {
    int n = v->get_n();
    vec rowmu = row_mean(v);
    vec colmu = col_mean(v);
    double mu = mean(v);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            v->set(i, j, v->at(i, j) - rowmu.at(i) - colmu.at(j) + mu);
        }
    }
}

mat matrix_sign(mat* v) {
    int n = v->get_n();
    mat result(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double val = v->at(i, j);
            if (val > 0) {
                result.set(i, j, 1);
            } else if (val < 0) {
                result.set(i, j, -1);
            } else {
                result.set(i, j, 0);
            }
        }
    }
    return result;
}

vec vec_sign(vec* v) {
    int n = v->get_n();
    vec result(n);
    for (int i = 0; i < n; i++) {
        double val = v->at(i);
        if (val > 0) {
            result.set(i, 1);
        } else if (val < 0) {
            result.set(i, -1);
        } else {
            result.set(i, 0);
        }
    }
    return result;
}

vec vec_sinh(vec* v) {
    int n = v->get_n();
    vec result(n);
    for (int i = 0; i < n; i++) {
        result.set(i, sinh(v->at(i)));
    }
    return result;
}

vec vec_cosh(vec* v) {
    int n = v->get_n();
    vec result(n);
    for (int i = 0; i < n; i++) {
        result.set(i, cosh(v->at(i)));
    }
    return result;
}

vec veh_tanh(vec* v) {
    int n = v->get_n();
    vec result(n);
    for (int i = 0; i < n; i++) {
        result.set(i, tanh(v->at(i)));
    }
    return result;
}

void vec_sort(vec* v){
    int n = v->get_n();
    std::sort(v->get_data(), v->get_data() + n);
}

void sort_by(vec* v1, vec* v2) {
    // Sort v1 and rearrange v2 accordingly
    int n = v1->get_n();
    vec temp1 = vec(*v1);
    vec temp2 = vec(*v2);
    vec_sort(v1);
    for (int i = 0; i < n; i++) {
        int index = temp1.find(v1->at(i));
        v2->set(i, temp2.at(index));
    }
}

double kendalls_tau(vec* v1, vec* v2) {
    // Compute's Kendall's Tau
    int n = v1->get_n();
    int concordant = 0;
    int discordant = 0;
    for (int i = 0; i < n; i++) {
        double x1 = v1->at(i);
        double y1 = v2->at(i);
        for (int j = i + 1; j < n; j++) {
            double x2 = v1->at(j);
            double y2 = v2->at(j);
            if ((x1 < x2 && y1 < y2) || (x1 > x2 && y1 > y2)) {
                concordant++;
            } else if ((x1 < x2 && y1 > y2) || (x1 > x2 && y1 < y2)) {
                discordant++;
            }
        }
    }
    return (concordant - discordant) / (0.5 * n * (n - 1));
}