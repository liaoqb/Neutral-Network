#include "Mnist.h"
#include <fstream>
#include <iostream>

int Mnist::reverseToInt(int number) {
  unsigned char c1 = number & 0xff;
  unsigned char c2 = (number >> 8) & 0xff;
  unsigned char c3 = (number >> 16) & 0xff;
  unsigned char c4 = (number >> 24) & 0xff;

  return int(c1 << 24) + int(c2 << 16) + int(c3 << 8) + int(c4);
}

void Mnist::readData() {
  std::ifstream imageFile(imageFileName, std::ifstream::binary);
  std::ifstream labelFile(labelFileName, std::ifstream::binary);

  if (imageFile.is_open()) {
	int magicNumber;
	int numberOfImage;
	int nRows;
	int nCols;

	imageFile.read((char*)&magicNumber, sizeof(magicNumber));
	imageFile.read((char*)&numberOfImage, sizeof(numberOfImage));
	row = reverseToInt(numberOfImage);
	imageFile.read((char*)&nRows, sizeof(nRows));
	nRows = reverseToInt(nRows);
	imageFile.read((char*)&nCols, sizeof(nCols));
	nCols = reverseToInt(nCols);
	col = nRows * nCols;

	// take care of this
	//row = 6000;
	images = new double*[row];

	for (int i = 0; i < row; ++i) {
	  images[i] = new double[col];
	}

	// read all the image data file and normalization
	for (int k = 0; k < row; ++k) {
	  for (int i = 0; i < nRows; ++i) {
		for (int j = 0 ; j < nCols; ++j) {
		  unsigned char temp = 0;
		  imageFile.read((char*)&temp, sizeof(temp));
		  images[k][i * nRows + j] = (temp == 0 ? 0 : 1);
		}
	  }
	}

	imageFile.close();
  }

  if (labelFile.is_open()) {
	int magicNumber;
	int items;

	labelFile.read((char*)&magicNumber, sizeof(magicNumber));
	labelFile.read((char*)&items, sizeof(items));

	items = reverseToInt(items);

	  // take care of this
	//items = 6000;

	labels = new int[items];

	for (int i = 0; i < items; ++i) {
	  unsigned char temp = 0;
	  labelFile.read((char*)&temp, sizeof(temp));
	  labels[i] = temp;
	}

	labelFile.close();
  }
}

Mnist::~Mnist() {
  if (images != NULL) {
	for (int i = 0; i < row; ++i) {
	  if (images[i] != NULL) {
	    delete images[i];
	  }
	}

	delete images;
  }

  if (labels != NULL) {
    delete labels;
  }

  images = NULL;
  labels = NULL;
}