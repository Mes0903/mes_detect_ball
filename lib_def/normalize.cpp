#include "normalize.h"

#include <iostream>
#include <Eigen/Eigen>
#include <fstream>

void Normalizer::fit(const Eigen::MatrixXd &data)
{
  const int COLS = data.cols();
  data_min = Eigen::VectorXd::Zero(COLS);
  data_mm = Eigen::VectorXd::Zero(COLS);

  for (int i = 0; i < COLS; ++i)
  {
    data_min(i) = data.col(i).minCoeff();
    data_mm(i) = data.col(i).maxCoeff() - data_min(i);
    if (data_mm(i) == 0)
      data_mm(i) = 1;
  }
}

Eigen::MatrixXd Normalizer::transform(const Eigen::MatrixXd &data)
{
  Eigen::MatrixXd tf_matrix(data.rows(), data.cols());

  const int COLS = data.cols();
  for (int i = 0; i < COLS; ++i)
    tf_matrix.col(i) = (data.col(i).array() - data_min(i)) / data_mm(i);

  return tf_matrix;
}

void Normalizer::store_weight([[maybe_unused]] const std::string filepath, std::ofstream &outfile)
{
  outfile << data_min.size() << ' ' << data_mm.size() << '\n';
  for (int i = 0; i < data_min.size(); ++i)
    outfile << data_min(i) << " \n"[i == data_min.size() - 1];

  for (int i = 0; i < data_mm.size(); ++i)
    outfile << data_mm(i) << " \n"[i == data_mm.size() - 1];

  std::cout << "Successfly stored normalizer!\n";
}

void Normalizer::load_weight([[maybe_unused]] const std::string filepath, std::ifstream &infile)
{
  std::string line;
  std::stringstream stream;

  int min_size, mm_size;
  getline(infile, line);
  stream << line;
  stream >> min_size >> mm_size;
  stream.str("");
  stream.clear();
  data_min = Eigen::VectorXd::Zero(min_size);
  data_mm = Eigen::VectorXd::Zero(mm_size);

  getline(infile, line);
  stream << line;
  for (int i = 0; i < min_size; ++i)
    stream >> data_min(i);

  stream.str("");
  stream.clear();

  getline(infile, line);
  stream << line;
  for (int i = 0; i < mm_size; ++i)
    stream >> data_mm(i);
}