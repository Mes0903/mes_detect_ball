#pragma once
#ifndef __LOAD_MATRIX_H__
#define __LOAD_MATRIX_H__

#include <Eigen/Eigen>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>

Eigen::MatrixXd readDataSet(const char *filename, const uint32_t ROWS, const uint32_t COLS)
{
  std::ifstream infile(filename);
  if (infile.fail())
  {
    std::cout << "cant found " << filename << '\n';
    exit(1);
  }

  Eigen::MatrixXd result(ROWS, COLS);
  std::string line;
  std::stringstream stream;
  uint32_t row = 0;
  while (!infile.eof())
  {
    double buff;
    getline(infile, line);
    stream << line;
    for (uint32_t col = 0; col < COLS; ++col)
    {
      stream >> buff;
      result(row, col) = buff;
    }
    ++row;

    stream.str("");
    stream.clear();
  }

  infile.close();
  return result;
};

Eigen::VectorXd readLabel(const char *filename, const uint32_t SIZE)
{
  std::ifstream infile(filename);
  if (infile.fail())
  {
    std::cout << "cant found " << filename << '\n';
    exit(1);
  }

  Eigen::VectorXd result(SIZE);
  std::string line;
  std::stringstream stream;

  uint32_t row = 0;
  while (!infile.eof())
  {
    double buff;
    getline(infile, line);
    stream << line;
    stream >> buff;
    result(row) = buff;
    ++row;

    stream.str("");
    stream.clear();
  }

  infile.close();
  return result;
}

#endif