# Definir el compilador
CXX = g++

# Definir cualquier bandera de compilación
CXXFLAGS = -Wall -std=c++11

# Definir cualquier directorio que contenga encabezados
INCLUDES = -I.

# Definir cualquier biblioteca que necesites enlazar
LDFLAGS = 

# Definir cualquier directorio que contenga bibliotecas
LDPATH = 

# Definir el nombre del archivo ejecutable
EXEC = mi_programa

# Definir los archivos objeto
OBJS = AlgoritmoBL.o AlgoritmoRELIEF.o Clasificador1NN.o DatasetHandler.o main.o RandomGenerator.o

# Regla para enlazar el ejecutable
$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(EXEC) $(OBJS) $(LDFLAGS) $(LDPATH)

# Regla para compilar los archivos fuente a objetos
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Definir los archivos de cabecera
DEPS = AlgoritmoBL.h AlgoritmoRELIEF.h Clasificador1NN.h DatasetHandler.h RandomGenerator.h

# Incluir las dependencias
-include $(DEPS:.h=.d)

# Regla para las dependencias
%.d: %.cpp
	$(CXX) -M $(CXXFLAGS) $< > $@

# Regla para limpiar los archivos no necesarios
clean:
	rm -f $(OBJS) $(EXEC) $(DEPS:.h=.d)

# Regla para limpiar todo, incluso los ejecutables
distclean: clean
	rm -f *~ $(EXEC)

# Regla pseudo-objetivo para evitar conflictos con archivos del mismo nombre
.PHONY: clean distclean
