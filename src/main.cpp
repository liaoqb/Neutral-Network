#include <iostream>
#include "Mnist.h"
#include "Layer.h"
#include "Network.h"

using namespace std;

int main(int argc, char* argv[]) {
  int train = 5;
  double rate = 0.01;
  int batch = 1;

  for (int i = 1; i < argc; ++i) {
	string t = "-t";
	string r = "-r";
	string b = "-b";

	if (strcmp(argv[i], t.c_str()) == 0) {
	  train = atoi(argv[i + 1]);
	}

	if (strcmp(argv[i], r.c_str()) == 0) {
	  rate = atof(argv[i + 1]);
	}

	if (strcmp(argv[i], b.c_str()) == 0) {
	  batch = atoi(argv[i + 1]);
	}
  }

  Mnist mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte");

  double** images = mnist.getImages();
  int* labels = mnist.getLabels();

  Network network(images, mnist.getRow(), mnist.getCol(), labels, 10);
  
  network.addLayer(Layer(mnist.getCol(), 300));
  network.addLayer(Layer(300, 75));
  network.addLayer(Layer(75, 10));

  network.readData("data/result.txt");

  network.train(train, rate, batch);

  network.saveData("data/result.txt");

  Mnist test("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
  network.predicate(test.getImages(), test.getRow(), test.getLabels());

  return 0;
}
