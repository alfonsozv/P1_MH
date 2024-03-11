# Práctica 1: Problema APC

## Autor

- **Nombre:** Alfonso Julián Zapata Velasco
- **DNI:** 77389094M
- **Correo electrónico:** [alfonsozv@correo.ugr.es](mailto:alfonsozv@correo.ugr.es)
- **Grupo de prácticas:** MH (A1)
- **Curso académico:** 23/24

## Índice

1. [Descripción del problema](#descripción-del-problema)
2. [Aplicación de los algoritmos](#aplicación-de-los-algoritmos)
   - [Algoritmo básico 1-NN](#algoritmo-básico-1-nn)
3. [Método de búsqueda y operaciones relevantes](#método-de-búsqueda-y-operaciones-relevantes)
4. [Algoritmos de comparación](#algoritmos-de-comparación)
5. [Procedimiento de desarrollo & README](#procedimiento-de-desarrollo--readme)
6. [Experimentos y análisis de resultados](#experimentos-y-análisis-de-resultados)
7. [Referencias bibliográficas](#referencias-bibliográficas)

## Descripción del problema

El problema del Aprendizaje de Pesos en Características (APC) se focaliza en optimizar el rendimiento de clasificadores basados en el método de los k Vecinos Más Cercanos (k-NN), con especial énfasis en el clasificador 1-NN, que elige la clase de la instancia más cercana para realizar la predicción. La optimización se realiza a través de la asignación de pesos a las características de las instancias, afectando directamente cómo se calculan las distancias entre ellas. La meta es maximizar tanto la precisión del clasificador (`tasa_clas`) como la simplicidad del modelo (`tasa_red`), equilibrando así el rendimiento con la complejidad del clasificador. Este equilibrio se gestiona mediante la función objetivo `F(W) = alpha * tasa_clas(W) + (1 - alpha) * tasa_red(W)`, donde `alpha` es un parámetro que pondera la importancia relativa entre precisión y simplicidad.

### Aplicación de los algoritmos

#### Algoritmo básico 1-NN

El 1-NN es uno de los algoritmos más simples de aprendizaje supervisado. Dado un conjunto de datos de entrenamiento, con etiquetas conocidas, el algoritmo clasifica nuevas instancias, basándose en la similitud con las instancias del conjunto de entrenamiento.
Para una nueva instancia, el algoritmo busca en el conjunto de entrenamiento, la instancia más cercana (“el vecino más cercano”) y asigna la etiqueta de esa instancia más cercana a la nueva instancia.

Para el desarrollo de este algoritmo he seguido la siguiente estructura: 

Representación de datos ⇒ 

Definimos una estructura para representar un conjunto de datos. Para una mayor velocidad, utilizaremos la librería Armadillo, para realizar operaciones de álgebra lineal. En esta representación, utilizaremos una matriz de datos, donde cada fila es una instancia y cada columna es una característica y un vector de etiquetas, donde cada elemento es la etiqueta de la instancia correspondiente.

Cálculo de la distancia ⇒ 

La distancia euclidiana se utiliza para encontrar el punto más cercano (o “vecino más cercano”) a un punto dado dentro de un conjunto de datos. El punto con la menor distancia euclidiana, se considerará el punto más cercano al punto de consulta.

Parámetros de la función: const arma::rowvec& a y const arma::rowvec& b: Los parámetros a y b son referencias constantes a objetos de tipo arma::rowvec, que representan vectores fila en Armadillo. Estos vectores fila, contienen las coordenadas de dos puntos entre los cuales se calculará la distancia euclidiana. Utilizar referencias constantes asegura que los vectores no se copien innecesariamente al llamar a la función, mejorando la eficiencia, y que no sean modificados dentro de la función.
Cuerpo de la función : arma::norm(a - b, 2): Esta línea es el núcleo de la función. Aquí es donde Armadillo realiza el cálculo de la distancia Euclidiana.
a - b: Primero, se calcula la diferencia entre los vectores a y b, resultando en un nuevo vector que representa la diferencia coordenada a 
// coordenada entre los dos puntos. 
// arma::norm(...): Luego, se calcula la norma (o longitud) de este vector de diferencia usando la función norm. El segundo argumento de la función norm, 
// que en este caso es 2, especifica que se debe calcular la norma L2, también conocida como norma Euclidiana. 


Algoritmo K-NN ⇒ 

El algoritmo K-NN (K-Nearest Neighbors) es un algoritmo de clasificación simple que se basa en la idea de que los puntos similares, deben tener etiquetas similares. Dado un conjunto de datos de entrenamiento con etiquetas conocidas, el algoritmo clasifica nuevas instancias basándose en la similitud con las instancias del conjunto de entrenamiento. Para una nueva instancia, el algoritmo busca en el conjunto de entrenamiento los k puntos más cercanos (los "vecinos más cercanos") y asigna la etiqueta más común entre esos k puntos a la nueva instancia.

Parámetros de la función: const arma::mat& trainingFeatures: El parámetro trainingFeatures es una referencia constante a un objeto de tipo arma::mat, que representa la matriz de datos de entrenamiento. Cada fila de esta matriz es una instancia y cada columna es una característica. const arma::Row<int>& trainingLabels: El parámetro trainingLabels es una referencia constante a un objeto de tipo arma::Row<int>, que representa el vector de etiquetas de entrenamiento. Cada elemento de este vector es la etiqueta de la instancia correspondiente en la matriz de datos de entrenamiento. const arma::rowvec& testInstance: El parámetro testInstance es una referencia constante a un objeto de tipo arma::rowvec, que representa una instancia de prueba para la cual se desea predecir la etiqueta.
