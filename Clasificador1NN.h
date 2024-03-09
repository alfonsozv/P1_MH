#ifndef CLASIFICADOR1NN_H
#define CLASIFICADOR1NN_H

#include <armadillo>

///////////////////////////////////////////////////////////////
/////////////// Representación de datos////////////////////////
///////////////////////////////////////////////////////////////

// Definimos una estructura para representar un conjunto de datos. 
// Para una mayor velocidad, utilizaremos la librería Armadillo para 
// realizar operaciones de álgebra lineal.


// Matriz de datos --> Cada fila es una instancia y cada columna es una característica
// Vector de etiquetas --> Cada elemento es la etiqueta de la instancia correspondiente

struct Dataset {
    arma::mat data; // Matriz de datos
    arma::Col<int> labels; // Vector de etiquetas
};

// Función para calcular la distancia euclidiana
double euclideanDistance(const arma::rowvec& a, const arma::rowvec& b);

// Algoritmo 1-NN
int oneNN(const arma::mat& trainingFeatures, const arma::Row<int>& trainingLabels, const arma::rowvec& testInstance);

#endif // CLASIFICADOR1NN_H
