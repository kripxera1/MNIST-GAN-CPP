#ifndef ALGEBRA_H
#define ALGEBRA_H

#include <vector>
#include <cmath>
using namespace std;

vector<vector<double>> sqrt(const vector<vector<double>> & M);

vector<vector<double>> sumE(const vector<vector<double>> & M, double e);

vector<double> sumE(const vector<double> & v, double e);

vector<double> sqrt(const vector<double> & v);

vector<double> elemWiseDiv(const vector<double> & v1, const vector<double> & v2);

vector<vector<double>> sumM(const vector<vector<double>> & M1, const vector<vector<double>> & M2);

vector<double> sumV(const vector<double> & v1, const vector<double> & v2);

vector<double> hadamard(const vector<double> & v1, const vector<double> & v2);

vector<double> minusV(const vector<double> & v1, const vector<double> & v2);

vector<double> rowsSum(const vector<vector<double>> & M);




vector<double> product(vector<double> v, double a);




vector<vector<double>> product(vector<vector<double>> M, double a);




vector<vector<double>> T(const vector<vector<double>>&M);




vector<vector<double>> dot(const vector<vector<double>> & M1,
                           const vector<vector<double>> & M2);



vector<vector<double>> sum(const vector<vector<double>> & M,
                           const vector<double> & b);



vector<vector<double>> minusM(const vector<vector<double>> & M1,
                              const vector<vector<double>> & M2);



vector<vector<double>> hadamard(const vector<vector<double>> & M1,
                                const vector<vector<double>> & M2);




vector<vector<double>> elemWiseDiv(const vector<vector<double>> & m1,
                                const vector<vector<double>> & m2);

vector<vector<double>> concatenate(const vector<vector<double>> & m1,
                                    const vector<vector<double>> & m2);
#endif