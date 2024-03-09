#ifndef DATASETHANDLER_H
#define DATASETHANDLER_H

#include <string>
#include <armadillo> // Asegúrate de que la biblioteca Armadillo esté instalada

class DatasetHandler {
public:
    // Constructor que carga el dataset desde un archivo.
    DatasetHandler(const std::string& filename);

    // Destructor.
    ~DatasetHandler();

    // Carga los datos desde el archivo ARFF.
    bool loadArff(const std::string& filename);

    // Retorna una referencia a la matriz de datos.
    const arma::mat& getData() const;

    // Retorna una referencia al vector de etiquetas.
    const arma::Col<int>& getLabels() const;

    // Retorna el número de características (excluyendo la etiqueta).
    size_t getFeatureCount() const;

    // Retorna el número de instancias en el dataset.
    size_t getInstanceCount() const;

private:
    std::string filename; // Nombre del archivo del dataset.
    arma::mat data;       // Matriz de datos.
    arma::Col<int> labels; // Vector de etiquetas.
    size_t n_features;    // Número de características.
    size_t n_samples;     // Número de instancias de muestra.

    // Funciones auxiliares que podrías necesitar para procesar el archivo ARFF.
    void processArffLine(const std::string& line);
    void convertToArmadilloMatrix(const std::vector<std::vector<double>>& rawData,
                                  const std::vector<int>& rawLabels);
    void loadDataset();
};

#endif // DATASETHANDLER_H
