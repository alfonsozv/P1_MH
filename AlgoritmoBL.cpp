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

// Se empleará el movimiento de cambio por mutación normal para generar vecinos
arma::rowvec AlgoritmoBL::generarVecino(const arma::rowvec& W) {
    arma::rowvec nuevoW = W; // Copiar W para no modificar el original
    // Aplicar mutación normal a un elemento aleatorio de W
    int index = rand() % W.size(); // Elegir un índice aleatorio
    double mutation = arma::randn() * 0.3; // Usar varianza de 0.3 como ejemplo
    nuevoW[index] += mutation; // Aplicar la mutación al índice elegido

    // Asegurar que los pesos se mantienen en el rango [0, 1]
    nuevoW = arma::clamp(nuevoW, 0.0, 1.0); // Usar 0.0 y 1.0 como límites inferior y superior
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

// Ejecución del Algoritmo BL
void AlgoritmoBL::ejecutarBL() {
    bool mejora = true;
    int evaluaciones = 0;
    const int MAX_EVALUACIONES = 15000;
    const int MAX_VECINOS = 20 * dataset.data.n_cols; // 20*n
    int vecinosGenerados = 0; // Contador para los vecinos generados sin mejora

    // Exploración del vecindario: En cada paso de la exploraciión, se mutará un componente distinto sin repetición, en un orden aleatorio distinto para cada solución,
    // hasta que haya mejora o se hayan modificado todas las posiciones una vez sin conseguir mejora.
    // En ese momento se comienza una nueva exploración sobre la nueva solución aceptada, si ha habido mejora, o sobre la actual si no la ha habido.
    while (evaluaciones < MAX_EVALUACIONES && vecinosGenerados < MAX_VECINOS) {
        arma::rowvec vecino = generarVecino(this->dataset.W); 
        double valorObjetivoVecino = objectiveFunction(vecino);
        if (valorObjetivoVecino > objectiveFunction(this->dataset.W)) { // Criterio de aceptacion ==> Se considera una mejora cuando aumenta el valor global de la función objetivo
            this->dataset.W = vecino; // Aceptamos el nuevo vecino como solución actual
            mejora = true;
            vecinosGenerados = 0; // Restablecer el contador de vecinos ya que encontramos una mejora
        } else {
            vecinosGenerados++; // Incrementar el contador de vecinos generados sin mejora
        }
        evaluaciones++;
        
        // Si no hay mejora, se sigue iterando hasta alcanzar el máximo de vecinos generados
        if (vecinosGenerados >= MAX_VECINOS) {
            mejora = false; // Establecer mejora a falso para salir del bucle si se alcanza el máximo de vecinos sin mejora
        }
    }
}

