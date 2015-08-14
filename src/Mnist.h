  // <Copyright liaoqb>  [Copyright 2015.07.08]

#ifndef MNIST_H_
#define MNIST_H_

#include <string>
#include <vector>

class Mnist {
public:
  Mnist(std::string imageFileName, std::string labelFileName) :
	imageFileName(imageFileName), labelFileName(labelFileName) {
    row = 0;
	col= 0;
	images = NULL;
	labels = NULL;

	readData();
  }

  ~Mnist();
  int getRow() {return row;}
  int getCol() {return col;}
  
  double** getImages() {return images;}
  int* getLabels() {return labels;}

private:
  double** images;
  int* labels;
  std::string imageFileName;
  std::string labelFileName;
  int row;
  int col;

  void readData();
  int reverseToInt(int number);
};

#endif