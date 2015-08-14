#include"Layer.h"
#include <ctime>
#include <cstdlib>
#include <cmath>

Layer::Layer(const Layer& layer) {
  input = layer.input;
  output = layer.output;
  initialize();
}

void Layer::setBatch(int batch) {
  this ->batch = batch;

  values = new double*[batch];

  for (int i = 0; i < batch; ++i) {
    values[i] = new double[output];
  }
}

Layer::~Layer() {
  if (weights != NULL) {
	for (int i = 0; i < output; ++i) {
	  delete weights[i];
	}

	delete weights;
  }

  if (errors != NULL) {
    delete errors;
  }

  if (thresholds != NULL) {
	delete thresholds;
  }

  if (values != NULL) {
	for (int i = 0; i < batch; ++i) {
	  delete values[i];
	}

	delete values;
  }

  values = NULL;
  weights = NULL;
  errors = NULL;
  thresholds = NULL;
}

void Layer::initialize() {
    // take care
  values = NULL;
  srand(time(NULL));

  errors = new double[output];
  thresholds = new double[output];

  weights = new double*[output];

  for (int i = 0; i < output; ++i) {
	errors[i] = 0.0;
	thresholds[i] = rand() / RAND_MAX;

    weights[i] = new double[input];

	for (int j = 0; j < input; ++j) {
	  weights[i][j] = 2.0 * rand() / RAND_MAX - 1.0;
	}
  }
}

double Layer::sigmoid(double value) {
  return 1.0 / (1.0 + exp(-value));
}

void Layer::calculateValues(double** inputValues, int offset) {
  //std::cout << offset << std::endl;
  for (int k = 0; k < offset; ++k) {
    for (int i = 0; i < output; ++i) {
	  double all = 0.0;
	  for (int j = 0; j < input; ++j) {
	    all += weights[i][j] * inputValues[k][j];
	  }

	  values[k][i] = sigmoid(all - thresholds[i]);

	//std::cout << averigeValues[i] << std::endl;
    }
  }
}

void Layer::calculateErrors(double* inputErrors, double** inputWeights, int inputSize, int offset) {
  double* temp = new double[output];
  clearErrors();

  for (int k = 0; k < offset; ++k) {
	for (int i = 0; i < output; ++i) {
	  temp[i] = 0.0;
	  for (int j = 0; j < inputSize; ++j) {
		temp[i] += inputErrors[j] * inputWeights[j][i];
	  }

	  temp[i] *= values[k][i] * (1 - values[k][i]);
	  errors[i] += temp[i];
	}
  }

  for (int i = 0; i < output; ++i) {
    errors[i] /= double(offset);
  }

  delete temp;
  temp = NULL;
}

void Layer::calculateWeights(double rate, double** inputValues, int offset) {
  double** temp = new double*[output];

  for (int i = 0; i < output; ++i) {
    temp[i] = new double[input];
	
	for (int j = 0; j < input; ++j) {
	  temp[i][j] = 0.0;
	}
  }

  for (int k = 0; k < offset; ++k) {
	for (int i = 0; i < input; ++i) {
	  for (int j = 0; j < output; ++j) {
		temp[j][i] += rate * errors[j] * inputValues[k][i];
	  }
	}
  }

  for (int i = 0; i < input; ++i) {
	for (int j = 0; j < output; ++j) {
	  weights[j][i] += temp[j][i] / double(offset);
	}
  }

  for (int i = 0; i < output; ++i) {
    delete temp[i];
  }

  delete temp;
  temp = NULL;
}

void Layer::clearErrors() {
  for (int i = 0; i < output; ++i) {
    errors[i] = 0.0;
  }
}