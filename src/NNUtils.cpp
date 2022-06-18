#include "NNUtils.h"
#include <iostream>

vector<vector<double>> normM(const vector<vector<double>> m, double mean, double std){
    vector<vector<double>> resultado(m.size(),
    vector<double>(m[0].size(),0));
    for(int i = 0; i < m.size(); i++){
        for(int j = 0; j < m[0].size(); j++){
            resultado[i][j] = (m[i][j]-mean)/std;
        }
    }
    return resultado;
}

vector<vector<double>> bceDeriv(const vector<vector<double>> & Y,
                                const vector<vector<double>> & A){
    auto resultado = product(minusM(elemWiseDiv(Y,A),elemWiseDiv(minusM(vector<vector<double>>(Y.size(),vector<double>(Y[0].size(),1)),Y),minusM(vector<vector<double>>(A.size(),vector<double>(A[0].size(),1)),A))),-1);
    return resultado;
}

double sigmoid(double input){
    return(1/(1+exp(-input)));
}

//sigmoid derivative with respect to activation
double sigmoidDeriv(double input){
    return(input*(1-input));
}


vector<vector<double>> sigmoid(const vector<vector<double>> & input){
    vector<vector<double>> resultado(input.size(),
    vector<double>(input[0].size(),0));

    for(int i = 0; i < input.size(); i++){
        for(int j = 0; j < input[i].size(); j++){
            resultado[i][j]=sigmoid(input[i][j]);
        }
    }
    return resultado;

}

vector<vector<double>> sigmoidDeriv(const vector<vector<double>> & input){
    vector<vector<double>> resultado(input.size(),
    vector<double>(input[0].size(),0));

    for(int i = 0; i < input.size(); i++){
        for(int j = 0; j < input[i].size(); j++){
            resultado[i][j]=sigmoidDeriv(input[i][j]);
        }
    }
    return resultado;

}

vector<vector<double>> tanh(const vector<vector<double>> & input){
    vector<vector<double>> resultado(input.size(),
    vector<double>(input[0].size(),0));


    for(int i = 0; i < input.size(); i++){
        for(int j = 0; j < input[i].size(); j++){
            resultado[i][j]=tanh(input[i][j]);
        }
    }
    return resultado;
}

vector<vector<double>> tanhDeriv(const vector<vector<double>> & input){
    vector<vector<double>> resultado(input.size(),
    vector<double>(input[0].size(),0));


    for(int i = 0; i < input.size(); i++){
        for(int j = 0; j < input[i].size(); j++){
            resultado[i][j]=1-(tanh(input[i][j])*tanh(input[i][j]));
        }
    }
    return resultado;
}



  
void normalDist(vector<vector<double>> & matrix,
                        double mean, double stdDev){
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(mean,stdDev);

    for(int i = 0; i < matrix.size(); i++)
        for(int j = 0; j < matrix[i].size(); j++)
            matrix[i][j] = distribution(generator);

}


void softMax(vector<double> & input){

	double m, Z, constant;

	m = -INFINITY;
	for (int i = 0; i < input.size(); i++)
		if (m < input[i]) 
			m = input[i];
		
	Z = 0.0;
	for (int i = 0; i < input.size(); i++) 
		Z += exp(input[i] - m);

	constant = m + log(Z);
	for (int i = 0; i < input.size(); i++) 
		input[i] = exp(input[i] - constant);

}


vector<vector<double>> softMax(const vector<vector<double>> & input){

    vector<vector<double>> A(input.size(),vector<double>(input[0].size()));
    vector<double> aux(input.size(),0);
    auto aux2 = aux;
    for(int j = 0; j < input[0].size(); j++){
        for(int i = 0; i < input.size(); i++)
            aux[i]=input[i][j];
        softMax(aux);
        for(int i = 0; i < aux.size(); i++)
            A[i][j] = aux[i];
    }

    return A;
}


vector<vector<double>> leakyRelu(const vector<vector<double>> & z){

    vector<vector<double>> A(z.size(),vector<double>(z[0].size()));
    for(int i = 0; i < z.size(); i++)
        for(int j = 0; j < z[0].size(); j++)
            A[i][j] = z[i][j] >= 0 ? z[i][j]:(0.2*z[i][j]);

    return A;
}



vector<vector<double>> leakyReluDeriv(const vector<vector<double>> & z){

    vector<vector<double>> A(z.size(),vector<double>(z[0].size()));
    for(int i = 0; i < z.size(); i++)
        for(int j = 0; j < z[0].size(); j++)
            A[i][j] = z[i][j] >= 0 ? 1 : 0.2;

    return A;
}


void initWeightsBias(vector<vector<double>> & W, vector<double> & b){
    
    for(int i = 0; i < W.size();i++){
        b[i]=(((double)rand()/RAND_MAX)-0.5)*0.5;
        for(int j = 0; j < W[0].size(); j++)
            W[i][j]=(((double)rand()/RAND_MAX)-0.5)*0.5;
        }
}


void updateWeightsBias(vector<vector<double>> & W,
                   const vector<vector<double>> & dW, vector<double> &b,
                   const vector<double> &db, double learnRate){

    for(int i = 0; i < b.size(); i++)
        b[i]-= db[i]*learnRate;
    for(int i = 0; i < W.size(); i++)
        for(int j = 0; j < W[0].size(); j++)
            W[i][j]-=dW[i][j]*learnRate;
}


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
                   double eps){

    vector<vector<double>> mWHat(W.size(),vector<double>(W[0].size(),0));
    vector<vector<double>> vWHat(W.size(),vector<double>(W[0].size(),0));
    vector<double> mbHat(mb.size(),0);
    vector<double> vbHat(vb.size(),0);

    //Weights
    mW = sumM(product(mW,beta1),product(dW,1.0-beta1));
    vW = sumM(product(vW,beta2),product(hadamard(dW,dW),1.0-beta2));
    mWHat = product(mW,1/(1.0-pow(beta1,iter+1)));
    vWHat = product(vW,1/(1.0-pow(beta2,iter+1)));
    W = minusM(W,elemWiseDiv(product(mWHat,learnRate),sumE(sqrt(vWHat),eps)));
    //biases
    mb = sumV(product(mb,beta1),product(db,1.0-beta1));
    vb = sumV(product(vb,beta2),product(hadamard(db,db),1.0-beta2));
    mbHat = product(mb,1/(1.0-pow(beta1,iter+1)));
    vbHat = product(vb,1/(1.0-pow(beta2,iter+1)));
    b = minusV(b,elemWiseDiv(product(mbHat,learnRate),sumE(sqrt(vbHat),eps)));

}
   
vector<vector<int>> loadData(const char* fileName){

    ifstream file(fileName,ios::in);
    string line;
    vector<vector<int>> data;
    while (getline(file, line)) {
        istringstream is(line);
        data.push_back(vector<int>(istream_iterator<int>(is)
        ,istream_iterator<int>()));
    }
    return data;
}


vector<vector<double>> loadBatch(const vector<vector<int>> & data,
                                 int batchSize, int it){

    vector<vector<double>> A(batchSize,vector<double>(data[0].size()-1,0));
    for(int i = batchSize*it,k=0; i < batchSize*(it+1); i++,k++)
        for(int j = 1; j < data[0].size(); j++)
            A[k][j]=(double)data[i][j]/255;

    A=T(A);

    return A;
}


vector<vector<double>> oneHotEncoding(const vector<vector<int>> & data,
                                      int batchSize, int it){

    vector<vector<double>> oneHot(10,vector<double>(batchSize,0));
    vector<double> labels(batchSize,0);

    for(int i = batchSize*it, k = 0; i < (it+1)*batchSize;k++, i++)
        labels[k] = data[i][0];
    for(int i = 0; i < labels.size();i++)
        oneHot[labels[i]][i] = 1;
    
    return oneHot;
}


vector<int> getPrediction(const vector<vector<double>> & A){

    vector<int> v;
    for(int j = 0; j < A[0].size(); j++){
        int maxInt;
        double max = -INFINITY;
        for(int i = 0; i < A.size(); i++)
            if(A[i][j]>max){
                max = A[i][j];
                maxInt = i;                
            }
        v.push_back(maxInt);
    }

    return v;
}


double accuracy(vector<vector<double>>A, vector<vector<double>>Y){

    double correctos = 0;
    for(int j = 0; j < A[0].size(); j++){
        int maxInt;
        double max = -INFINITY;
        for(int i = 0; i < A.size(); i++)
            if(A[i][j]>max){
                max = A[i][j];
                maxInt = i;
            }
        if(Y[maxInt][j]==1)
            correctos+=1;
    }

    return correctos/Y[0].size();
}

void print(vector<vector<double>> a){
    a = T(a);
    for(int i = 0; i < a.size(); i++){
        for(int j = 0; j < a[i].size(); j++){
            cout << a[i][j] << " ";
        }
        cout << endl;
    }
}

vector<vector<double>> average(const vector<vector<double>> & m1, const vector<vector<double>> & m2){
    vector<vector<double>> resultado(m1.size(),
    vector<double>(m1[0].size(),0));

    for(int i = 0; i < m1.size(); i++){
        for(int j = 0; j < m1[0].size(); j++){
            resultado[i][j]=(m1[i][j]+m2[i][j])/2;
        }
    }

    return resultado;
}

vector<double> average(const vector<double> & v1, const vector<double> & v2){
    vector<double>resultado(v1.size(),0);
    
    for(int i = 0; i < v1.size(); i++){
        resultado[i]=(v1[i]+v2[i])/2;

    }

    return resultado;
}