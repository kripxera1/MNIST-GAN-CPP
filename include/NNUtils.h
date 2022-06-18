#ifndef NNUTILS_H
#define NNUTILS_H

#include <fstream>
#include <sstream>
#include <iterator>
#include <vector>
#include <cmath>
#include <random>
#include "algebra.h"

using namespace std;

void adamOptimizer(vector<vector<double>> & W,
                   const vector<vector<double>> & dW,
                   vector<vector<double>> & vW,
                   vector<vector<double>> & mW,
                   vector<double> &b,
                   const vector<double> &db,
                   vector<double> & vb,
                   vector<double> & mb,
                   double learnRate,
                   double beta1,
                   double beta2,
                   int iter,
                   double eps = 0.001);
vector<double> average(const vector<double> & v1, const vector<double> & v2);



vector<vector<double>> average(const vector<vector<double>> & m1, const vector<vector<double>> & m2);


vector<vector<double>> normM(const vector<vector<double>> m, double mean, double std);


//binary cross entropy loss derivative with respect to activation
vector<vector<double>> bceDeriv(const vector<vector<double>> & Y,
                                const vector<vector<double>> & A);


vector<vector<double>> sigmoidDeriv(const vector<vector<double>> & input);



vector<vector<double>> sigmoid(const vector<vector<double>> & input);



void softMax(vector<double> & input);



void normalDist(vector<vector<double>> & matrix,
                        double mean, double stdDev);


vector<vector<double>> tanh(const vector<vector<double>> & input);


vector<vector<double>> tanhDeriv(const vector<vector<double>> & input);


vector<vector<double>> softMax(const vector<vector<double>> & input);



vector<vector<double>> leakyRelu(const vector<vector<double>> & z);



vector<vector<double>> leakyReluDeriv(const vector<vector<double>> & z);



void initWeightsBias(vector<vector<double>> & W, vector<double> & b);



void updateWeightsBias(vector<vector<double>> & W,
                       const vector<vector<double>> & dW, vector<double> &b,
                       const vector<double> &db, double learnRate);


vector<vector<int>> loadData(const char* fileName);




vector<vector<double>> loadBatch(const vector<vector<int>> & data,
                                 int batchSize, int it);



vector<vector<double>> oneHotEncoding(const vector<vector<int>> & data,
                                      int batchSize, int it);



vector<int> getPrediction(const vector<vector<double>> & A);



double accuracy(vector<vector<double>>A, vector<vector<double>>Y);




void print(vector<vector<double>>  a);
#endif