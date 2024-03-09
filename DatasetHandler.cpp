
// En DatasetHandler.cpp se encuentran las definiciones para el manejo y conversión de los datos en formato .arff 
// a un formato que pueda ser utilizado por el algoritmo de clasificación.

// Incluimos Armadillo para las operaciones de álgebra lineal
#include <armadillo>
#include "DatasetHandler.h"
#include <fstream>
#include <sstream>
#include <string>

// Constructor de la clase DatasetHandler


DatasetHandler::DatasetHandler(const std::string& filename) {
    this->filename = filename;
    this->data = new arma::mat();
    this->labels = new arma::Col<int>();
    loadDataset();
}

void DatasetHandler::loadDataset() {
    std::ifstream file(this->filename);
    std::string line;
    std::vector<std::vector<double>> dataset;
    std::vector<int> labelset;
    
    // Leer archivo .arff línea por línea
    while (getline(file, line)) {
        // Ignorar líneas no útiles
        if (line.empty() || line[0] == '%' || line.substr(0, 10) == "@attribute" || line.substr(0, 5) == "@data") {
            continue;
        }
        
        // Procesar la línea
        std::istringstream iss(line);
        std::vector<double> row;
        double value;
        int label;
        int count = 0;
        
        // Leer cada valor separado por comas, asumiendo que la última columna es la etiqueta
        while (iss >> value) {
            if (iss.peek() == ',') iss.ignore();
            if (++count == n_features + 1) {
                label = static_cast<int>(value);
                labelset.push_back(label);
            } else {
                row.push_back(value);
            }
        }
        dataset.push_back(row);
    }

    // Convertir los vectores de datos y etiquetas en objetos de Armadillo
    *this->data = arma::mat(dataset.size(), n_features);
    *this->labels = arma::Col<int>(labelset.size());
    
    for (size_t i = 0; i < dataset.size(); ++i) {
        this->data->row(i) = arma::vec(dataset[i]);
        this->labels->at(i) = labelset[i];
    }
}

// ... Otras funciones de manejo de datos ...

DatasetHandler::~DatasetHandler() {
    delete this->data;
    delete this->labels;
}


