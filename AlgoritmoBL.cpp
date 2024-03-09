// AlgoritmoBL.cpp
#include "AlgoritmoBL.h"
#include <armadillo>
#include <cmath>
#include <limits>


///////////////////////////////////////////////////////////////
////////////// Constructor de la clase/////////////////////////
///////////////////////////////////////////////////////////////

AlgoritmoBL::AlgoritmoBL(const arma::mat& data, const arma::Col<int>& labels) {
    this->dataset.data = data;
    this->dataset.labels = labels;
    this->dataset.W = arma::randu<arma::rowvec>(data.n_cols); // Inicialización aleatoria de W (Generación de una solución inicial)
}

///////////////////////////////////////////////////////////////
///// Función para generar vecinos de una solución actual /////
///////////////////////////////////////////////////////////////

arma::rowvec AlgoritmoBL::generarVecino(const arma::rowvec& W, int indice) {
    arma::rowvec nuevoW = W;
    double mutation = arma::randn() * 0.3; // Varianza de 0.3
    nuevoW[indice] += mutation; // Aplicar la mutación en la característica específica
    nuevoW = arma::clamp(nuevoW, 0.0, 1.0); // Mantener los pesos en [0, 1]
    return nuevoW;
}




    
double AlgoritmoBL::calcularPrecision(const arma::mat& data, const arma::Col<int>& labels, const arma::rowvec& W) {
    int correctos = 0; // Contador para instancias bien clasificadas

    // Iteramos sobre todas las instancias usando leave-one-out
    for (size_t i = 0; i < data.n_rows; ++i) {
        // Creamos el conjunto de entrenamiento excluyendo la instancia actual
        arma::mat trainingSet = arma::join_vert(data.rows(0, i - 1), data.rows(i + 1, data.n_rows - 1));
        arma::Col<int> trainingLabels = arma::join_vert(labels.rows(0, i - 1), labels.rows(i + 1, labels.n_rows - 1));

        // Ponderamos las características por los pesos
        arma::mat weightedTrainingSet = trainingSet.each_row() % W;
        arma::rowvec weightedTestInstance = data.row(i) % W;

        // Calculamos las distancias entre la instancia de prueba y todas las instancias de entrenamiento
        arma::vec distances = arma::sqrt(arma::sum(arma::pow(weightedTrainingSet.each_row() - weightedTestInstance, 2), 1));

        // Encontramos el índice del vecino más cercano
        arma::uword indexNearest;
        distances.min(indexNearest);

        // Comprobamos si la clasificación es correcta
        if (trainingLabels[indexNearest] == labels[i]) {
            ++correctos; // Incrementamos el contador si la clasificación es correcta
        }
    }

    // Calculamos la precisión como el porcentaje de clasificaciones correctas
    return 100.0 * correctos / data.n_rows;
}

// Función para calcular la complejidad de W
double calcularComplejidad(const arma::rowvec& W) {
    // Umbral para considerar que una característica ha sido descartada
    const double umbral = 0.1;

    // Contamos el número de pesos que son menores que el umbral
    int caracteristicasDescartadas = arma::accu(W < umbral);

    // Calculamos la tasa de reducción como el porcentaje de características descartadas
    double tasa_red = 100.0 * caracteristicasDescartadas / W.n_elem;

    return tasa_red;
}


// Función objetivo
double AlgoritmoBL::objectiveFunction(const arma::rowvec& W) {
    // Calcular la precisión y la complejidad basada en W
    double accuracy = calcularPrecision(dataset.data, dataset.labels, W);
    double complexity = calcularComplejidad(W);
    
    // Maximizar F(W) = α·tasa_clas(W) + (1-α)·tasa_red(W)
    // Nota: La tasa de reducción ya está en términos de "simplicidad" (mayor es mejor),
    // por lo que no necesitamos (1 - complexity)
    return alpha * accuracy + (1 - alpha) * complexity;
}

// Algoritmo de Búsqueda Local
void AlgoritmoBL::ejecutarBL() {
    bool mejora = true;
    int evaluaciones = 0;
    const int MAX_EVALUACIONES = 15000; // Número máximo de evaluaciones de la función objetivo
    int n_caracteristicas = dataset.data.n_cols; // Número de características
    const int MAX_VECINOS = 20 * n_caracteristicas; // Número máximo de vecinos a explorar
    std::vector<int> indices(n_caracteristicas); // Vector para seguir las características ya mutadas
    std::iota(indices.begin(), indices.end(), 0); // Llenar el vector con valores de 0 a n-1

    // Búsqueda local

    while (mejora && evaluaciones < MAX_EVALUACIONES) { // Condición de parada ==> mejora = false o evaluaciones >= MAX_EVALUACIONES
        mejora = false;
        std::random_shuffle(indices.begin(), indices.end()); // Orden aleatorio sin repetición
        for (int i = 0; i < MAX_VECINOS && !mejora; ++i) { // Condición de parada ==> mejora = true
            arma::rowvec vecino = generarVecino(this->dataset.W, indices[i % n_caracteristicas]); // Generar un vecino
            double valorObjetivoVecino = objectiveFunction(vecino); // Calcular el valor objetivo del vecino
            if (valorObjetivoVecino > objectiveFunction(this->dataset.W)) {
                this->dataset.W = vecino; // Aceptamos el nuevo vecino como solución actual
                mejora = true;
                std::iota(indices.begin(), indices.end(), 0); // Reiniciar los índices para la próxima iteración
            }
            evaluaciones++;
        }
    }
}
