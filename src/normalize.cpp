#include "normalize.h"

#include <Eigen/Eigen>

void Normalizer::fit(const Eigen::MatrixXd &data)
{
  const int COLS = data.cols();
  data_min = VectorXd::Zero(COLS);
  data_mm = VectorXd::Zero(COLS);

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

  for (int i = 0; i < COLS; ++i)
    tf_matrix.col(i) = (data.col(i).array() - data_min(i)) / data_mm(i);

  return tf_matrix;
}