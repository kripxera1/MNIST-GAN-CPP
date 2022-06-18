#include "algebra.h"
#include <iostream>



vector<double> minusV(const vector<double> & v1, const vector<double> & v2){
    auto resultado = v1;
    for(int i = 0; i < v2.size(); i++){
        resultado[i] -= v2[i];
    }
    return resultado;
}

//Pre: number of columns of firs equals number of rows of second
vector<double> elemWiseDiv(const vector<double> & v1, const vector<double> & v2){
    auto resultado = v1;
    for(int i = 0; i < v2.size(); i++){
        resultado[i]/=v2[i];
    }
    return resultado;
}

vector<double> sumE(const vector<double> & v, double e){
    auto resultado = v;
    for(int i = 0; i < v.size(); i++){
        resultado[i] += e;
    }
    return resultado;
}

vector<double> sqrt(const vector<double> & v){
    auto resultado = v;
    for(int i = 0; i < v.size(); i++){
        resultado[i]=sqrt(resultado[i]);
    }
    return resultado;
}

vector<double> hadamard(const vector<double> & v1, const vector<double> & v2){
    auto resultado = v1;
    for(int i = 0; i < v2.size(); i++){
        resultado[i]+=v2[i];
    }
    return resultado;
}
vector<double> sumV(const vector<double> & v1, const vector<double> & v2){
    auto resultado = v1;
    for(int i = 0; i < v2.size(); i++){
        resultado[i]+=v2[i];
    }
    return resultado;
}


vector<vector<double>> sumE(const vector<vector<double>> & M, double e){
    auto resultado=M;
    for(int i = 0; i < M.size(); i++){
        for(int j = 0; j < M[0].size();j++){
            resultado[i][j]+=e;
        }
    }
    return resultado;
}


vector<vector<double>> sqrt(const vector<vector<double>> & M){
     auto resultado=M;
    for(int i = 0; i < M.size(); i++){
        for(int j = 0; j < M[0].size();j++){
            resultado[i][j]=sqrt(resultado[i][j]);
        }
    }
    return resultado;
}

vector<vector<double>> dot(const vector<vector<double>> & M1,
                           const vector<vector<double>> & M2){

    vector<vector<double>>M3(M1.size(),vector<double>(M2[0].size(),0));
    for(int i = 0; i < M1.size(); i++)
        for(int j = 0; j < M2[0].size(); j++)
            for(int k = 0; k < M2.size(); k++)
                M3[i][j] += M1[i][k]*M2[k][j];

    return M3;
}

vector<vector<double>> sumM(const vector<vector<double>> & M1,
 const vector<vector<double>> & M2){
    auto resultado = M1;
    for(int i = 0; i < M2.size(); i++){
        for(int j = 0; j < M2[0].size();j++){
            resultado[i][j]+=M2[i][j];
        }
    }
    return resultado;
}

//sums vector b with each column of the matrix M
//Pre: M.size() == b.size();
vector<vector<double>> sum(const vector<vector<double>> & M,
                           const vector<double> & b){

    vector<vector<double>> M2(M.size(),vector<double>(M[0].size(),0));
    for(int i = 0; i < M.size(); i++)
        for(int j = 0; j < M[0].size(); j++)
           M2[i][j]=b[i]+M[i][j];

    return M2;
}


vector<vector<double>> T(const vector<vector<double>>&m){

    vector<vector<double>> mt(m[0].size(),vector<double>(m.size(),0));
    for(int i = 0; i < m.size(); i++)
        for(int j = 0; j < m[0].size(); j++)
            mt[j][i] = m[i][j];

    return mt;
}


vector<vector<double>> minusM(const vector<vector<double>> & m1,
                             const vector<vector<double>> & m2){

    vector<vector<double>> m(m1.size(),vector<double>(m1[0].size(),0));
    for(int i = 0; i < m1.size(); i++)
        for(int j = 0; j < m1[0].size(); j++)
            m[i][j]=m1[i][j]-m2[i][j];

    return m;
}


vector<vector<double>> product(vector<vector<double>> m1, double a){

    for(int i = 0; i < m1.size(); i++)
        for(int j = 0; j < m1[0].size(); j++)
            m1[i][j]*=a;

    return m1;
}


vector<double> product(vector<double> v, double a){

    for(int i = 0; i<v.size(); i++)
        v[i]*=a;

    return v;
}

//devuelve un vector con la suma de cada una de las filas
vector<double> rowsSum(const vector<vector<double>> & m){

    vector<double> v (m.size(), 0);
    for(int i = 0; i < m.size(); i++)
        for(int j = 0; j< m[i].size(); j++)
            v[i]+=m[i][j];
    
    return v;
}


vector<vector<double>> hadamard(const vector<vector<double>> & m1,
                                const vector<vector<double>> & m2){

    vector<vector<double>> m(m1.size(),vector<double>(m1[0].size(),0));
    for(int i = 0; i < m1.size(); i++)
        for(int j = 0; j < m1[0].size(); j++)
            m[i][j]=m1[i][j]*m2[i][j];

    return m;
}


vector<vector<double>> elemWiseDiv(const vector<vector<double>> & m1,
                                const vector<vector<double>> & m2){

    vector<vector<double>> m(m1.size(),vector<double>(m1[0].size(),0));
    for(int i = 0; i < m1.size(); i++)
        for(int j = 0; j < m1[0].size(); j++){
            m[i][j]=m1[i][j]/m2[i][j];
            if(m1[i][j]== 0 && m2[i][j] == 0){
                cout << "indeterminaci" << endl;
                m[i][j] = 0.25;
            }
        }

    return m;


}


vector<vector<double>> concatenate(const vector<vector<double>> & m1,
                                    const vector<vector<double>> & m2){
    vector<vector<double>> resultado (m1.size(),
    vector<double>(m1[0].size()+m2[0].size(),0));
    
    for(int i = 0; i < m1.size(); i++){
        for(int j = 0; j < m1[0].size()+m2[0].size();j++){
            resultado[i][j]=j<m1[0].size()?m1[i][j]:m2[i][j-m1[0].size()];
        }
    }
    return resultado;
}


vector<double> concatenate (const vector<double> & v1, const vector<double> & v2){
    vector<double> resultado(v1.size()+v2.size(),0);
    for(int j = 0; j < v1.size()+v2.size();j++){
        resultado[j]=j<v1.size()?v1[j]:v2[j-v1.size()];
    }

    return resultado;
}