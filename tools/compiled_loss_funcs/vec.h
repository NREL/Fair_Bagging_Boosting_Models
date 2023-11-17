#include <cmath>

#pragma once

using namespace std;

// vec class with constructor and destructor
class vec {
    private:
        int n;
        double* data;

    public:
        // Constructor
        vec(int n) {
            this->n = n;
            this->data = new double[n];
        }
        vec(int n, double* data) {
            this->n = n;
            this->data = new double[n];
            for (int i = 0; i < n; i++) {
                this->data[i] = data[i];
            }
        }
        vec(int n, double val) {
            this->n = n;
            this->data = new double[n];
            for (int i = 0; i < n; i++) {
                this->data[i] = val;
            }
        }

        // Destructor
        ~vec() {
            delete[] this->data;
        }
        // Copy constructor
        vec(const vec& other) {
            this->n = other.n;
            this->data = new double[this->n];
            for (int i = 0; i < this->n; i++) {
                this->data[i] = other.data[i];
            }
        }
        // Copy assignment
        vec& operator=(const vec& other) {
            if (this != &other) {
                delete[] this->data;
                this->n = other.n;
                this->data = new double[this->n];
                for (int i = 0; i < this->n; i++) {
                    this->data[i] = other.data[i];
                }
            }
            return *this;
        }

        // Getters
        int get_n() {
            return this->n;
        }
        double* get_data() {
            return this->data;
        }
        double at(int i) {
            return this->data[i];
        }
        vec get_slice(int start, int end) {
            vec output(end - start);
            for (int i = start; i < end; i++) {
                output.set(i - start, this->data[i]);
            }
            return output;
        }

        // Setters
        void set(int i, double val) {
            this->data[i] = val;
        }
        void set_all(double val) {
            for (int i = 0; i < this->n; i++) {
                this->data[i] = val;
            }
        }
        void set_abs() {
            for (int i = 0; i < this->n; i++) {
                this->data[i] = abs(this->data[i]);
            }
        }
        void set_slice(int start, int end, vec other) {
            for (int i = start; i < end; i++) {
                this->data[i] = other[i - start];
            }
        }

        // Find 
        int find(double val) {
            for (int i = 0; i < this->n; i++) {
                if (this->data[i] == val) {
                    return i;
                }
            }
            return -1;
        }

        // Operators
        double operator[](int i) {
            return this->data[i];
        }

        // Addition
        vec operator+(vec other) {
            vec output(this->n);
            for (int i = 0; i < this->n; i++) {
                output.set(i, this->data[i] + other[i]);
            }
            return output;
        }
        void operator+=(vec other) {
            for (int i = 0; i < this->n; i++) {
                this->data[i] += other[i];
            }
        }
        vec operator+(double val) {
            vec output(this->n);
            for (int i = 0; i < this->n; i++) {
                output.set(i, this->data[i] + val);
            }
            return output;
        }
        void operator+=(double val) {
            for (int i = 0; i < this->n; i++) {
                this->data[i] += val;
            }
        }

        // Subtraction
        vec operator-(vec other) {
            vec output(this->n);
            for (int i = 0; i < this->n; i++) {
                output.set(i, this->data[i] - other[i]);
            }
            return output;
        }
        void operator-=(vec other) {
            for (int i = 0; i < this->n; i++) {
                this->data[i] -= other[i];
            }
        }
        vec operator-(double val) {
            vec output(this->n);
            for (int i = 0; i < this->n; i++) {
                output.set(i, this->data[i] - val);
            }
            return output;
        }
        void operator-=(double val) {
            for (int i = 0; i < this->n; i++) {
                this->data[i] -= val;
            }
        }

        // Multiplication
        vec operator*(vec other) {
            vec output(this->n);
            for (int i = 0; i < this->n; i++) {
                output.set(i, this->data[i] * other[i]);
            }
            return output;
        }
        void operator*=(vec other) {
            for (int i = 0; i < this->n; i++) {
                this->data[i] *= other[i];
            }
        }
        vec operator*(double val) {
            vec output(this->n);
            for (int i = 0; i < this->n; i++) {
                output.set(i, this->data[i] * val);
            }
            return output;
        }
        void operator*=(double val) {
            for (int i = 0; i < this->n; i++) {
                this->data[i] *= val;
            }
        }

        // Division
        vec operator/(vec other) {
            vec output(this->n);
            for (int i = 0; i < this->n; i++) {
                output.set(i, this->data[i] / other[i]);
            }
            return output;
        }
        void operator/=(vec other) {
            for (int i = 0; i < this->n; i++) {
                this->data[i] /= other[i];
            }
        }
        vec operator/(double val) {
            vec output(this->n);
            for (int i = 0; i < this->n; i++) {
                output.set(i, this->data[i] / val);
            }
            return output;
        }
        void operator/=(double val) {
            for (int i = 0; i < this->n; i++) {
                this->data[i] /= val;
            }
        }

        // Exponentiation
        vec operator^(int exp) {
            vec output(this->n);
            for (int i = 0; i < this->n; i++) {
                output.set(i, pow(this->data[i], exp));
            }
            return output;
        }
        void operator^=(int exp) {
            for (int i = 0; i < this->n; i++) {
                this->data[i] = pow(this->data[i], exp);
            }
        }
        vec operator^(double exp) {
            vec output(this->n);
            for (int i = 0; i < this->n; i++) {
                output.set(i, pow(this->data[i], exp));
            }
            return output;
        }
        void operator^=(double exp) {
            for (int i = 0; i < this->n; i++) {
                this->data[i] = pow(this->data[i], exp);
            }
        }
};