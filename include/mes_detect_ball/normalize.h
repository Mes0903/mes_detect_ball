#ifndef __NORMALIZER
#define __NORMALIZER

#include <Eigen/Eigen>

class Normalizer
{
public:
  Eigen::VectorXd data_min;
  Eigen::VectorXd data_mm;

public:
  void fit(const Eigen::MatrixXd &data);
  Eigen::MatrixXd transform(const Eigen::MatrixXd &x);
};

#endif