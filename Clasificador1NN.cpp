// Código del clasificador 1-NN (Clasificador1NN.cpp):

// El 1-NN es uno de los algoritmos más simples de aprendizaje supervisado. 
// Dado un conjunto de datos de entrenamiento con etiquetas conocidas, el algoritmo clasifica nuevas 
// instancias basándose en la similitud con las instancias del conjunto de entrenamiento. 
// Para una nueva instancia, el algoritmo busca en el conjunto de entrenamiento la instancia
// más cercana (el "vecino más cercano") y asigna la etiqueta de esa instancia más cercana a la nueva instancia.

#include <armadillo>
#include <iostream>


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

///////////////////////////////////////////////////////////////
/////////////// Cálculo de la distancia ///////////////////////
///////////////////////////////////////////////////////////////

// La distancia euclidiana se utiliza para encontrar el punto más cercano 
// (o "vecino más cercano") a un punto dado dentro de un conjunto de datos. 
// El punto con la menor distancia euclidiana se considera el más similar o más 
// cercano al punto de consulta.

// Parámetros de la Función
// const arma::rowvec& a y const arma::rowvec& b: Los parámetros a y b son referencias constantes a objetos de tipo arma::rowvec,
// que representan vectores fila en Armadillo. Estos vectores fila contienen las coordenadas de dos puntos entre los cuales se calculará 
// la distancia Euclidiana. Utilizar referencias constantes asegura que los vectores no se copien innecesariamente al llamar a la función, 
// mejorando la eficiencia, y que no sean modificados dentro de la función.

// Cuerpo de la Función
// arma::norm(a - b, 2): Esta línea es el núcleo de la función. Aquí es donde Armadillo realiza el cálculo de la distancia Euclidiana.
// a - b: Primero, se calcula la diferencia entre los vectores a y b, resultando en un nuevo vector que representa la diferencia coordenada a 
// coordenada entre los dos puntos. 
// arma::norm(...): Luego, se calcula la norma (o longitud) de este vector de diferencia usando la función norm. El segundo argumento de la función norm, 
// que en este caso es 2, especifica que se debe calcular la norma L2, también conocida como norma Euclidiana. 

double euclideanDistance(const arma::rowvec& a, const arma::rowvec& b) {
    return arma::norm(a - b, 2); // '2' indica la norma L2 (Euclidiana) Implementación muy eficiente de la biblioteca Armadillo
}


///////////////////////////////////////////////////////////////
//////////////////// Algoritmo k-NN ///////////////////////////
///////////////////////////////////////////////////////////////

// El algoritmo k-NN (k-Nearest Neighbors) es un algoritmo de clasificación simple que se basa en la idea de que los puntos
// similares deben tener etiquetas similares. Dado un conjunto de datos de entrenamiento con etiquetas conocidas, el algoritmo
// clasifica nuevas instancias basándose en la similitud con las instancias del conjunto de entrenamiento. Para una nueva instancia,
// el algoritmo busca en el conjunto de entrenamiento los k puntos más cercanos (los "vecinos más cercanos") y asigna la etiqueta
// más común entre esos k puntos a la nueva instancia.

// Parámetros de la Función
// const arma::mat& trainingFeatures: El parámetro trainingFeatures es una referencia constante a un objeto de tipo arma::mat, que representa
// la matriz de datos de entrenamiento. Cada fila de esta matriz es una instancia y cada columna es una característica.
// const arma::Row<int>& trainingLabels: El parámetro trainingLabels es una referencia constante a un objeto de tipo arma::Row<int>, que representa 
// el vector de etiquetas de entrenamiento. Cada elemento de este vector es la etiqueta de la instancia correspondiente en la matriz de datos de entrenamiento.
// const arma::rowvec& testInstance: El parámetro testInstance es una referencia constante a un objeto de tipo arma::rowvec, que representa una instancia de prueba
// para la cual se desea predecir la etiqueta.


int oneNN(const arma::mat& trainingFeatures, const arma::Row<int>& trainingLabels, const arma::rowvec& testInstance) {
    double minDistance = std::numeric_limits<double>::max(); // Inicializamos la distancia mínima con un valor muy grande
    int label = -1; // Inicializamos la etiqueta con un valor inválido (Las etiquetas son 0 o 1)

    // Iteramos sobre todas las instancias de entrenamiento
    for(size_t i = 0; i < trainingFeatures.n_rows; ++i) {
        double distance = euclideanDistance(trainingFeatures.row(i), testInstance); // Calculamos la distancia entre la instancia de prueba y la instancia de entrenamiento i
        if(distance < minDistance) { // Si la distancia es menor que la distancia mínima actual
            minDistance = distance; // Actualizamos la distancia mínima
            label = trainingLabels(i); // Actualizamos la etiqueta
        }
    }

    return label;
}
