#include "weak_learner.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <fstream>
#include <sstream>
#include <Eigen/Eigen>

std::pair<Eigen::VectorXd, double>
weak_learner::fit(const Eigen::MatrixXd &train_X, const Eigen::MatrixXd &train_Y,
                  const Eigen::MatrixXd &train_weight, uint32_t Iterations)
{
  uint32_t D = train_X.cols();

  static std::random_device rd;
  static std::default_random_engine gen(rd());
  std::normal_distribution<> dis(0, std::sqrt(D + 1));
  w = Eigen::VectorXd::NullaryExpr(D, [&]()
                                   { return dis(gen); });

  w0 = dis(gen);
  Eigen::VectorXd w_momentum = Eigen::VectorXd::Zero(D);
  double w0_momentum = 0.0;
  double alpha = 0.25;

  for (uint32_t i = 0; i < Iterations; ++i)
  {
    double lr = alpha / (1 + i / 10);

    Eigen::ArrayXd hx = (train_X * w).array() + w0;
    hx = logistic(hx);
    Eigen::VectorXd tmp = train_weight.array() * (train_Y.array() - hx);

    Eigen::VectorXd w_grad = train_X.transpose() * tmp;
    double w0_grad = tmp.sum();

    w_momentum = (w_momentum + lr * w_grad) * 0.9;
    w0_momentum = (w0_momentum + lr * w0_grad) * 0.9;
    w += w_momentum + lr * w_grad;
    w0 += w0_momentum + lr * w0_grad;
  }

  Eigen::VectorXd pred_Y = get_label(train_X);

  double err = 0.0;
  for (int i = 0; i < pred_Y.size(); ++i)
  {
    if (pred_Y(i) != train_Y(i))
    {
      err -= train_weight(i);
    }
    else
    {
      err += train_weight(i);
    }
  }

  return {pred_Y, err};
}

double weak_learner::logistic(double x)
{
  return (std::tanh(x / 2) + 1) / 2;
}

Eigen::ArrayXd weak_learner::logistic(Eigen::ArrayXd &x)
{
  return ((x / 2).tanh() + 1) / 2;
}

Eigen::VectorXd weak_learner::predict(const Eigen::MatrixXd &test_X) // test X is Sn*5
{
  Eigen::ArrayXd hx = (test_X * w).array() + w0;

  return logistic(hx);
}

Eigen::VectorXd weak_learner::get_label(const Eigen::MatrixXd &test_X)
{

  Eigen::ArrayXd hx = (test_X * w).array() + w0;

  return logistic(hx).round();
}

void weak_learner::store_weight(std::ofstream &outfile)
{
  uint32_t N = w.size();
  outfile << w0 << '\n';
  for (uint32_t i = 0; i < N; ++i)
    outfile << w(i) << " \n"[i == N - 1];
}

void weak_learner::load_weight(std::ifstream &infile, std::stringstream &stream)
{
  int N = 5;
  w = Eigen::VectorXd::Zero(N);
  std::string line;

  getline(infile, line);
  stream << line;
  stream >> w0;
  stream.str("");
  stream.clear();

  getline(infile, line);
  stream << line;
  for (int i = 0; i < N; ++i)
  {
    double buff;
    stream >> buff;
    w(i) = buff;
  }
}