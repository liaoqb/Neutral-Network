#include "Network.h"
#include <cmath>
#include <fstream>

Network::Network(double** inputs, int row, int col, int* outputs, int label) {
  this ->inputs = inputs;
  this ->outputs = outputs;
  this ->row = row;
  this ->col = col;
  this ->label = label;
}

void Network::addLayer(Layer layer) {
  layers.push_back(layer);
}

void Network::train(int times, double rate, int batch) {
  for (int i = 0; i < layers.size(); ++i) {
	layers[i].setBatch(batch);
  }

  for (int k = 0; k < times; ++k) {
	double allErrors = 0.0;
	  
	  // the loop times
	int loop = int(row / batch) + (row % batch == 0 ? 0 : 1);

	for (int i = 0; i < loop; ++i) {
	  int offset = ((i + 1) * batch > row ? row % batch : batch);
	  layers[0].calculateValues((double**)(&inputs[i * batch]), offset);
	  
	    // fp
	  for (int j = 1; j < layers.size(); ++j) {
		layers[j].calculateValues(layers[j - 1].getValues(), offset);
	  }

	  double* outputErrors = layers[layers.size() - 1].getErrors();
	  double** outputValues = layers[layers.size() - 1].getValues();

	  double er = 0.0;

	  for (int j = 0; j < label; ++j) {
	    outputErrors[j] = 0.0;
	  }

	  for (int z = 0; z < offset; ++z) {
		for (int j = 0; j < label; ++j) {
		  outputErrors[j] += (j == outputs[i * batch + z] ? 1.0 - outputValues[z][j] : - outputValues[z][j]) *
			outputValues[z][j] * (1.0 - outputValues[z][j]);

		  //std::cout << outputValues[z][j] << std::endl;

		  er += pow((j == outputs[i * batch + z] ? outputValues[z][j] - 1.0 : outputValues[z][j]), 2);
		}
	  }

	  for (int j = 0; j < label; ++j) {
		outputErrors[j] /= double(offset);
	  }

	  er /= 2.0;
	  allErrors += er;

	    // bp, this is not good
	  for (int j = layers.size() - 1; j >= 1; --j) {
		layers[j - 1].calculateErrors(layers[j].getErrors(),
		  layers[j].getWeights(), layers[j].getOutput(), offset);
	  }

	  for (int j = layers.size() - 1; j >= 1; --j) {
	    layers[j].calculateWeights(rate, layers[j - 1].getValues(), offset);
	  }
	    // This is a problem
	  layers[0].calculateWeights(rate, (double**)(&inputs[i * batch]), offset);

	  for (int x = 0; x < layers.size(); ++x) {
		double* thresholds = layers[x].getThresholds();
		double* errors = layers[x].getErrors();

		for (int y = 0; y < layers[x].getOutput(); ++y) {		  
		  thresholds[y] += rate * errors[y];
		  //std::cout << thresholds[y] << std::endl;
		}
	  }
	}

	std::cout << allErrors << std::endl;
  }
}

void Network::predicate(double** images, int size, int* labels) {
  std::ofstream fout;
  int count = 0;

  fout.open("data/label.txt", std::ios::out);

  for (int i = 0; i < layers.size(); ++i) {
	layers[i].setBatch(1);
  }

  for (int i = 0; i < size; ++i) {
	layers[0].calculateValues((double**)(&images[i]), 1);
	for (int j = 1; j < layers.size(); ++j) {
	  layers[j].calculateValues(layers[j - 1].getValues(), 1);
	}

	double** values = layers[layers.size() - 1].getValues();
	int index = 0;
	double maxi = values[0][0];

	for (int j = 0; j < label; ++j) {
	  //std::cout << values[j] << std::endl;
	  if (values[0][j] > maxi) {
	    maxi = values[0][j];
		index = j;
	  }
	}

	fout << index << std::endl;

	//std::cout << index << std::endl;
	//std::cout << std::endl;
	if (index == labels[i]) {
	  count++;
	}
  }

  std::cout << count / double(size) << std::endl;

  fout.close();
}

void Network::saveData(std::string fileName) {
  std::ofstream fout;
  fout.open(fileName, std::ios::out);

  if (fout.is_open()) {
    fout << layers.size() << std::endl;

	for (int i = 0; i < layers.size(); ++i) {
	  fout << layers[i].getOutput() << ' ' << layers[i].getInput() << std::endl;

	  double** weights = layers[i].getWeights();
	  double* thresholds = layers[i].getThresholds();

	  for (int j = 0 ; j < layers[i].getOutput(); ++j) {
		for (int k = 0; k < layers[i].getInput(); ++k) {
		  if (k) {
		    fout << ' ' << weights[j][k];
		  } else {
			fout << weights[j][k];
		  }
		}
		fout << std::endl;
	  }

	  for (int j = 0; j < layers[i].getOutput(); ++j) {
		if (j) {
		  fout << ' ' << thresholds[j];
		} else {
		  fout << thresholds[j];
		}
	  }

	  fout << std::endl;
	}

	fout.close();
	
	std::cout << "save data success!\n";
  } else {
    std::cout << "open file failed!\n";
  }
}

void Network::readData(std::string fileName) {
  std::ifstream fin;

  fin.open(fileName,std::ios::in);

  if (fin.is_open()) {
    int size;
	layers.clear();

	fin >> size;

	for (int i = 0; i < size; ++i) {
	  int output;
	  int input;

	  fin >> output >> input;

	  //std::cout << output << ' ' << input << std::endl;

	  layers.push_back(Layer(input, output));

	  double** weights = layers[layers.size() - 1].getWeights();

	  for (int j = 0; j < output; ++j) {
		for (int k = 0; k < input; ++k) {
		  fin >> weights[j][k];
		}
	  }
	  double* thresholds = layers[layers.size() - 1].getThresholds();

	  for (int j = 0; j < output; ++j) {
		fin >> thresholds[j];
	  }
	}

	fin.close();

	std::cout << "read data success!\n";
  } else {
    std::cout << "open file failed!\n";
  }
}