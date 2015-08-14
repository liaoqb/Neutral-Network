  // <Copyright liaoqb>  [Copyright 2015.07.09]
#ifndef NETWORK_H
#define NETWORK_H

#include "Layer.h"
#include <vector>

class Network {
public:
  Network(double** inputs, int row, int col, int* output, int label);
  void addLayer(Layer layer);
  void train(int times, double rate, int batch = 1);
  void predicate(double** images, int size, int* labels);
  ~Network() {}

  std::vector<Layer> getLayers() {return layers;}

  void saveData(std::string fileName);
  void readData(std::string fileName);

private:
  int row;
  int col;
  int label;
  double** inputs;
  int* outputs;
  std::vector<Layer> layers;
};

#endif
