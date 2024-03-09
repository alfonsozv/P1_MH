#ifndef CLASIFICADOR1NN_H
#define CLASIFICADOR1NN_H

#include <armadillo>

// Representación de datos
struct Dataset {
    arma::mat data; // Matriz de datos
    arma::Col<int> labels; // Vector de etiquetas
};

// Función para calcular la distancia euclidiana
double euclideanDistance(const arma::rowvec& a, const arma::rowvec& b);

// Algoritmo 1-NN
int oneNN(const arma::mat& trainingFeatures, const arma::Row<int>& trainingLabels, const arma::rowvec& testInstance);

#endif // CLASIFICADOR1NN_H
