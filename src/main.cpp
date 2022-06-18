#include <iostream>
#include <random>
#include "bitmap.h"
#include "algebra.h"
#include "NNUtils.h"

using namespace std;

int main(){
  
    srand(time(NULL));

    //Hyperparameter
    double learnRateD = 0.1;
    double learnRateG = 0.06;

    int batchSize = 100;
    int nEpochs = 20000000;

    cout << "Hyperparameters:\n"
         << "\n\tLearning rate:\t\t" << learnRateG
         << "\n\tBatch size:\t\t" << batchSize
         << "\n\tNumber of epochs:\t" << nEpochs
         << "\n" << endl;

    //Data loading
    cout << "\nLoading training set..." << endl;
    auto trainingData=loadData("mnist_test.txt");
    cout << "\n\tDone" << endl;

    int trainSize = trainingData.size();
    int nBatch = trainSize/batchSize;

    //Layer sizes Generator;
    int sizeG0 = 64;
    int sizeG1 = 256;
    int sizeG2 = 784;

    //Layer sizes Discriminator
    int sizeD0 = sizeG2;
    int sizeD1 = 128;
    int sizeD2 = 1;

    //Image test variables
    int nImages = 10;
    int height = 28;
    int width  = 28;
    auto image = reserveSpaceImage(height,width);

    //fixed distribution generator for test
    //normal distribution is added to input
    vector<vector<double>> fixedDist(sizeG0,vector<double>(nImages, 0));
    normalDist(fixedDist,0,1);

    vector<vector<double>> WG1(sizeG1,vector<double>(sizeG0,0));
    vector<double> bG1(sizeG1,0);
    vector<vector<double>> WG2(sizeG2,vector<double>(sizeG1,0));
    vector<double> bG2(sizeG2,0);

    vector<vector<double>> WD1(sizeD1,vector<double>(sizeD0,0));
    vector<double> bD1(sizeD1,0);
    vector<vector<double>> WD2(sizeD2,vector<double>(sizeD1,0));
    vector<double> bD2(sizeD2,0);

    //Weights and biases initialization
    initWeightsBias(WG1,bG1);
    initWeightsBias(WG2,bG2);

    initWeightsBias(WD1,bD1);
    initWeightsBias(WD2,bD2);

    //Training

    cout << "\n\nTraining accuracy:\n" << endl;
    for(int epoch = 0; epoch < nEpochs; epoch++){
        cout << "Epoch " << epoch << endl;
        for(int it = 0; it < nBatch; it++){
            //Train Discrminator
            //normal random distribution is added to input
            vector<vector<double>> AG0(sizeG0,vector<double>(batchSize));
            normalDist(AG0,0,1);
            //Feed forward Generator
            auto ZG1 = sum(dot(WG1,AG0),bG1);
            auto AG1 = leakyRelu(ZG1);
            auto ZG2 = sum(dot(WG2,AG1),bG2);
            auto AG2 = tanh(ZG2);

            //Fake images
            auto ADF0 = AG2;
            auto ZDF1 = sum(dot(WD1,ADF0),bD1);
            auto ADF1 = leakyRelu(ZDF1);
            auto ZDF2 = sum(dot(WD2,ADF1),bD2);
            auto ADF2 = sigmoid(ZDF2);
            auto YF =vector<vector<double>>(sizeD2,vector<double>(batchSize,0));

            //Real images
            auto ADR0 = normM(loadBatch(trainingData,batchSize,it),0.5,0.5);
            auto ZDR1 = sum(dot(WD1,ADR0),bD1);
            auto ADR1 = leakyRelu(ZDR1);
            auto ZDR2 = sum(dot(WD2,ADR1),bD2);
            auto ADR2 = sigmoid(ZDR2);
            auto YR =vector<vector<double>>(sizeD2,vector<double>(batchSize,1));

            //Combined
            auto Y   = concatenate(YR,  YF  );
            auto ZD2 = concatenate(ZDR2,ZDF2);
            auto ZD1 = concatenate(ZDR1,ZDR1);
            auto AD2 = concatenate(ADR2,ADF2);
            auto AD1 = concatenate(ADR1,ADF1);
            auto AD0 = concatenate(ADR0,ADF0);
            double sumaF = 0;
            double sumaR = 0;
            for(int i = 0; i < Y[0].size(); i++){
                if(Y[0][i]==0)
                    sumaF += AD2[0][i];
                if(Y[0][i]==1)
                    sumaR += AD2[0][i];   
            }
            sumaF/=YF[0].size();
            sumaR/=YR[0].size();
            cout << "1\t" << sumaR << endl;
            cout << "0\t" << sumaF << endl;



            //Backpropagation Discriminator
            auto dZD2 = hadamard(bceDeriv(Y,AD2),sigmoidDeriv(AD2));
            auto dWD2 = product(dot(dZD2,T(AD1)),1.0/(batchSize*2));
            auto dbD2 = product(rowsSum(dZD2),1.0/(batchSize*2));

            auto dZD1 = hadamard(dot(T(WD2),dZD2),(leakyReluDeriv(ZD1)));
            auto dWD1 = product(dot(dZD1,T(AD0)),1.0/(batchSize*2));
            auto dbD1 = product(rowsSum(dZD1),1.0/(batchSize*2));
            updateWeightsBias(WD1,dWD1,bD1,dbD1,learnRateD);
            updateWeightsBias(WD2,dWD2,bD2,dbD2,learnRateD);
        

            


            //Train Generator

            auto YG =vector<vector<double>>(sizeD2,vector<double>(batchSize,1));

            //Backpropagation Generator
            dZD2 = hadamard(bceDeriv(YG,ADF2),sigmoidDeriv(ADF2));
            dWD2 = product(dot(dZD2,T(ADF1)),1.0/batchSize);
            dbD2 = product(rowsSum(dZD2),1.0/batchSize);

            dZD1 = hadamard(dot(T(WD2),dZD2),(leakyReluDeriv(ZDF1)));
            dWD1 = product(dot(dZD1,T(AG2)),1.0/batchSize);
            dbD1 = product(rowsSum(dZD1),1.0/batchSize);

            auto dZG2 = hadamard(dot(T(WD1),dZD1),(tanhDeriv(ZG2)));
            auto dWG2 = product(dot(dZG2,T(AG1)),1.0/batchSize);
            auto dbG2 = product(rowsSum(dZG2),1.0/batchSize);

            auto dZG1 = hadamard(dot(T(WG2),dZG2),(leakyReluDeriv(ZG1)));
            auto dWG1 = product(dot(dZG1,T(AG0)),1.0/batchSize);
            auto dbG1 = product(rowsSum(dZG1),1.0/batchSize);
            updateWeightsBias(WG1,dWG1,bG1,dbG1,learnRateG);
            updateWeightsBias(WG2,dWG2,bG2,dbG2,learnRateG);
            



            if(it%1 == 0){
                            //Test image generation
                auto AG0 = fixedDist;
                auto ZG1 = sum(dot(WG1,AG0),bG1);
                auto AG1 = leakyRelu(ZG1);
                auto ZG2 = sum(dot(WG2,AG1),bG2);
                auto AG2 = tanh(ZG2);

                for(int images = 0; images < nImages; images++){

                    vector<double> vA;
                    for(int i = 0; i < AG2.size(); i++)
                        vA.push_back(AG2[i][images]);

                    auto it = vA.begin();

                    for(int i = height-1; i >= 0; i--)
                        for(int j = 0; j < width; j++){
                            image[i][j][0] = (unsigned char)(int)((*it+1)*125.5);
                            image[i][j][1] = (unsigned char)(int)((*it+1)*125.5);
                            image[i][j][2] = (unsigned char)(int)((*it+1)*125.5);
                            it++;
                        }

                    generateBitmapImage(image,height,width,(char*)
                    ((string(to_string(images))+".bmp").c_str()));
                }
            }
        }
    }
    
    return 0;
}
