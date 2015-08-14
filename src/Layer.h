#ifndef LAYER_H_
#define LAYER_H_

#include <iostream>

class Layer {
public:
  Layer(int input, int output) : input(input), output(output) {
    weights = NULL;
	values = NULL;
	errors = NULL;
	thresholds = NULL;

	initialize();
  }

  Layer(const Layer& layer);

  void calculateValues(double** inputValues, int offset);
  void calculateErrors(double* inputErrors, double** inputWeights, int inputSize, int offset);
  void calculateWeights(double rate, double** inputValues, int offset);

  double** getWeights() {return weights;}
  double** getValues() {return values;}
  double* getErrors() {return errors;}
  double* getThresholds() {return thresholds;}
  int getInput() {return input;}
  int getOutput() {return output;}
  int getBatch() {return batch;}

  void setBatch(int batch);

  void clearErrors();

  ~Layer();

private:
  int input;
  int output;
  int batch;

  double** weights;
  double** values;
  double* errors;
  double* thresholds;

  void initialize();
  double sigmoid(double value);
};

#endif