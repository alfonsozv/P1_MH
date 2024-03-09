// AlgoritmoBL.h
#ifndef ALGORITMO_BL_H
#define ALGORITMO_BL_H

#include <armadillo>

///////////////////////////////////////////////////////////////
///////////// Representación de datos////////////////////////
///////////////////////////////////////////////////////////////

// Matriz de datos --> Cada fila es una instancia y cada columna es una característica
// Vector de etiquetas --> Cada elemento es la etiqueta de la instancia correspondiente
// Vector de pesos --> Cada elemento es el peso de la característica correspondiente

struct Dataset_BL {
    arma::mat data; // Matriz de datos
    arma::Col<int> labels; // Vector de etiquetas
    arma::rowvec W; // Vector de pesos
};

class AlgoritmoBL {
public:
    AlgoritmoBL(const arma::mat& data, const arma::Col<int>& labels);
    void ejecutarBL();

private:
    Dataset_BL dataset;
    double alpha = 0.8; // Importancia entre clasificación y reducción

    double euclideanDistance(const arma::rowvec& a, const arma::rowvec& b);
    double calcularPrecision(const arma::mat& data, const arma::Col<int>& labels, const arma::rowvec& W);
    double calcularComplejidad(const arma::rowvec& W);
    double objectiveFunction(const arma::rowvec& W);
    arma::rowvec generarVecino(const arma::rowvec& W, int indice);
};

#endif // ALGORITMO_BL_H
