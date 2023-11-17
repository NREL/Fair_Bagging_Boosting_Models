#include <cmath>
#include "vec.h"

#pragma once 

using namespace std;

// mat class with constructor and destructor

class mat {
    private:
        int n;
        double** data;

    public:
        // Constructor
        mat(int n) {
            this->n = n;
            this->data = new double*[n];
            for (int i = 0; i < n; i++) {
                this->data[i] = new double[n];
            }
        }
        mat(int n, double** data) {
            this->n = n;
            this->data = new double*[n];
            for (int i = 0; i < n; i++) {
                this->data[i] = new double[n];
                for (int j = 0; j < n; j++) {
                    this->data[i][j] = data[i][j];
                }
            }
        }
        mat(int n, double val) {
            this->n = n;
            this->data = new double*[n];
            for (int i = 0; i < n; i++) {
                this->data[i] = new double[n];
                for (int j = 0; j < n; j++) {
                    this->data[i][j] = val;
                }
            }
        }

        // Destructor
        ~mat() {
            for (int i = 0; i < this->n; i++) {
                delete[] this->data[i];
            }
            delete[] this->data;
        }
        // Copy constructor
        mat(const mat& other) {
            this->n = other.n;
            this->data = new double*[this->n];
            for (int i = 0; i < this->n; i++) {
                this->data[i] = new double[this->n];
                for (int j = 0; j < this->n; j++) {
                    this->data[i][j] = other.data[i][j];
                }
            }
        }
        // Copy assignment
        mat& operator=(const mat& other) {
            if (this != &other) {
                for (int i = 0; i < this->n; i++) {
                    delete[] this->data[i];
                }
                delete[] this->data;
                this->n = other.n;
                this->data = new double*[this->n];
                for (int i = 0; i < this->n; i++) {
                    this->data[i] = new double[this->n];
                    for (int j = 0; j < this->n; j++) {
                        this->data[i][j] = other.data[i][j];
                    }
                }
            }
            return *this;
        }

        // Getters
        int get_n() {
            return this->n;
        }
        double** get_data() {
            return this->data;
        }
        double at(int i, int j) {
            return this->data[i][j];
        }
        vec row(int i) {
            vec row(this->n);
            for (int j = 0; j < this->n; j++) {
                row.set(j, this->data[i][j]);
            }
            return row;
        }
        vec col(int j) {
            vec col(this->n);
            for (int i = 0; i < this->n; i++) {
                col.set(i, this->data[i][j]);
            }
            return col;
        }

        // Setters
        void set(int i, int j, double val) {
            this->data[i][j] = val;
        }
        void set_all(double val) {
            for (int i = 0; i < this->n; i++) {
                for (int j = 0; j < this->n; j++) {
                    this->data[i][j] = val;
                }
            }
        }
        void set_abs() {
            for (int i = 0; i < this->n; i++) {
                for (int j = 0; j < this->n; j++) {
                    this->data[i][j] = abs(this->data[i][j]);
                }
            }
        }
        void set_row(int i, double* row) {
            for (int j = 0; j < this->n; j++) {
                this->data[i][j] = row[j];
            }
        }
        void set_row(int i, vec row) {
            for (int j = 0; j < this->n; j++) {
                this->data[i][j] = row.at(j);
            }
        }
        void set_row(int i, double val) {
            for (int j = 0; j < this->n; j++) {
                this->data[i][j] = val;
            }
        }
        void set_col(int j, double* col) {
            for (int i = 0; i < this->n; i++) {
                this->data[i][j] = col[i];
            }
        }
        void set_col(int j, vec col) {
            for (int i = 0; i < this->n; i++) {
                this->data[i][j] = col.at(i);
            }
        }
        void set_col(int j, double val) {
            for (int i = 0; i < this->n; i++) {
                this->data[i][j] = val;
            }
        }

        // Adders
        void add_to_row(int i, double* row) {
            for (int j = 0; j < this->n; j++) {
                this->data[i][j] += row[j];
            }
        }
        void add_to_row(int i, vec row) {
            for (int j = 0; j < this->n; j++) {
                this->data[i][j] += row.at(j);
            }
        }
        void add_to_row(int i, double val) {
            for (int j = 0; j < this->n; j++) {
                this->data[i][j] += val;
            }
        }
        void add_to_col(int j, double* col) {
            for (int i = 0; i < this->n; i++) {
                this->data[i][j] += col[i];
            }
        }
        void add_to_col(int j, vec col) {
            for (int i = 0; i < this->n; i++) {
                this->data[i][j] += col.at(i);
            }
        }
        void add_to_col(int j, double val) {
            for (int i = 0; i < this->n; i++) {
                this->data[i][j] += val;
            }
        }

        // Operators 
        // Assignment
        double* operator[](int i) {
            return this->data[i];
        }

        // Addition
        mat operator+(mat other) {
            mat result(this->n);
            for (int i = 0; i < this->n; i++) {
                for (int j = 0; j < this->n; j++) {
                    result.set(i, j, this->data[i][j] + other.at(i, j));
                }
            }
            return result;
        }
        void operator+=(mat other) {
            for (int i = 0; i < this->n; i++) {
                for (int j = 0; j < this->n; j++) {
                    this->data[i][j] += other.at(i, j);
                }
            }
        }
        mat operator+(double val) {
            mat result(this->n);
            for (int i = 0; i < this->n; i++) {
                for (int j = 0; j < this->n; j++) {
                    result.set(i, j, this->data[i][j] + val);
                }
            }
            return result;
        }
        void operator+=(double val) {
            for (int i = 0; i < this->n; i++) {
                for (int j = 0; j < this->n; j++) {
                    this->data[i][j] += val;
                }
            }
        }

        // Subtraction
        mat operator-(mat other) {
            mat result(this->n);
            for (int i = 0; i < this->n; i++) {
                for (int j = 0; j < this->n; j++) {
                    result.set(i, j, this->data[i][j] - other.at(i, j));
                }
            }
            return result;
        }
        void operator-=(mat other) {
            for (int i = 0; i < this->n; i++) {
                for (int j = 0; j < this->n; j++) {
                    this->data[i][j] -= other.at(i, j);
                }
            }
        }
        mat operator-(double val) {
            mat result(this->n);
            for (int i = 0; i < this->n; i++) {
                for (int j = 0; j < this->n; j++) {
                    result.set(i, j, this->data[i][j] - val);
                }
            }
            return result;
        }
        void operator-=(double val) {
            for (int i = 0; i < this->n; i++) {
                for (int j = 0; j < this->n; j++) {
                    this->data[i][j] -= val;
                }
            }
        }

        // Multiplication
        mat operator*(mat other) {
            mat result(this->n);
            // Compute element-wise multiplication
            for (int i = 0; i < this->n; i++) {
                for (int j = 0; j < this->n; j++) {
                    result.set(i, j, this->data[i][j] * other.at(i, j));
                }
            }
            return result;
        }
        void operator*=(mat other) {
            // Compute element-wise multiplication
            for (int i = 0; i < this->n; i++) {
                for (int j = 0; j < this->n; j++) {
                    this->data[i][j] *= other.at(i, j);
                }
            }
        }
        mat operator*(double val) {
            mat result(this->n);
            for (int i = 0; i < this->n; i++) {
                for (int j = 0; j < this->n; j++) {
                    result.set(i, j, this->data[i][j] * val);
                }
            }
            return result;
        }
        void operator*=(double val) {
            for (int i = 0; i < this->n; i++) {
                for (int j = 0; j < this->n; j++) {
                    this->data[i][j] *= val;
                }
            }
        }

        // Division
        mat operator/(mat other) {
            mat result(this->n);
            for (int i = 0; i < this->n; i++) {
                for (int j = 0; j < this->n; j++) {
                    result.set(i, j, this->data[i][j] / other.at(i, j));
                }
            }
            return result;
        }
        void operator/=(mat other) {
            for (int i = 0; i < this->n; i++) {
                for (int j = 0; j < this->n; j++) {
                    this->data[i][j] /= other.at(i, j);
                }
            }
        }
        mat operator/(double val) {
            mat result(this->n);
            for (int i = 0; i < this->n; i++) {
                for (int j = 0; j < this->n; j++) {
                    result.set(i, j, this->data[i][j] / val);
                }
            }
            return result;
        }
        void operator/=(double val) {
            for (int i = 0; i < this->n; i++) {
                for (int j = 0; j < this->n; j++) {
                    this->data[i][j] /= val;
                }
            }
        }

        // Exponentiation
        mat operator^(int exp) {
            mat result(this->n);
            for (int i = 0; i < this->n; i++) {
                for (int j = 0; j < this->n; j++) {
                    result.set(i, j, pow(this->data[i][j], exp));
                }
            }
            return result;
        }
        void operator^=(int exp) {
            for (int i = 0; i < this->n; i++) {
                for (int j = 0; j < this->n; j++) {
                    this->data[i][j] = pow(this->data[i][j], exp);
                }
            }
        }
};