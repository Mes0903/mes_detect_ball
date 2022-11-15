#pragma once
#ifndef __WEAK_LEARNER
#define __WEAK_LEARNER

#include <vector>
#include <Eigen/Eigen>
#include <fstream>

class weak_learner
{
public:
  double w0;
  Eigen::VectorXd w;

public:
  std::pair<Eigen::VectorXd, double> fit(const Eigen::MatrixXd &train_X, const Eigen::MatrixXd &train_Y,
                                         const Eigen::MatrixXd &train_weight, uint32_t Iterations);

  double logistic(double x);
  Eigen::ArrayXd logistic(Eigen::ArrayXd &x);
  Eigen::VectorXd get_label(const Eigen::MatrixXd &test_X);
  Eigen::VectorXd predict(const Eigen::MatrixXd &test_X);

public:
  void store_weight(std::ofstream &outfile);
  void load_weight(std::ifstream &infile, std::stringstream &stream);
};

#endif