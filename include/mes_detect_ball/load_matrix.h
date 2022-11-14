#pragma once
#ifndef __LOAD_MATRIX_H__
#define __LOAD_MATRIX_H__

#include <Eigen/Eigen>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>

void transform_to_xy(Eigen::MatrixXd &data, const uint32_t ROWS)
{
  for (uint32_t i = 0; i < ROWS; i++)
  {
    const double theta = data(i, 0);
    const double r = data(i, 1);

    data(i, 0) = r * std::cos(theta); // x
    data(i, 1) = r * std::sin(theta); // y
  }
}

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

  transform_to_xy(result, ROWS);
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