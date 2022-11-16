#include "adaboost_classifier.h"
#include "normalize.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <Eigen/Eigen>

void Adaboost::fit(const Eigen::MatrixXd &train_X, const Eigen::VectorXd &train_Y)
{
  Eigen::VectorXd w = Eigen::VectorXd::Ones(train_X.rows());

  for (int i = 0; i < M; ++i)
  {
    w /= w.sum();
    const auto [pred_Y, err] = T[i].fit(train_X, train_Y, w, 1500);

    if (err == 1.0)
    {
      auto tmp = T[i];
      T.clear();
      T.push_back(std::move(tmp));
      M = 1;
      alpha = Eigen::VectorXd::Ones(M);
      break;
    }

    alpha(i) = std::log((1 + err) / (1 - err)) / 2;
    for (int r = 0; r < train_Y.size(); ++r)
    {
      if (train_Y(r) != pred_Y(r))
        w(r) *= std::exp(alpha(i));
      else
        w(r) *= std::exp(-alpha(i));
    }
  }
}

Eigen::VectorXd Adaboost::predict(const Eigen::MatrixXd &test_X)
{
  uint32_t R = test_X.rows();
  Eigen::ArrayXd C(R);
  for (int m = 0; m < M; ++m)
    C += alpha(m) * (2 * T[m].get_label(test_X).array() - 1);

  return C.unaryExpr([](double x)
                     { return double(x > 0); });
}

void Adaboost::store_weight([[maybe_unused]] const std::string filepath, std::ofstream &outfile)
{
  outfile << static_cast<double>(TP) / (TP + FN) << ' ' << TN << ' ' << TP << ' ' << FN << ' ' << FP << '\n';
  outfile << M << '\n';
  for (int i = 0; i < M; ++i)
    outfile << alpha(i) << " \n"[i == M - 1];

  for (int i = 0; i < M; ++i)
    T[i].store_weight(outfile);

  std::cout << "Successfly stored Adaboost weighting!\n";
}

void Adaboost::load_weight([[maybe_unused]] const std::string filepath, std::ifstream &infile)
{
  std::string line;
  std::stringstream stream;
  double correct_rate;

  getline(infile, line);
  stream << line;
  stream >> correct_rate >> TN >> TP >> FN >> FP;
  stream.str("");
  stream.clear();

  getline(infile, line);
  stream << line;
  stream >> M;
  stream.str("");
  stream.clear();

  alpha = Eigen::VectorXd::Zero(M);
  getline(infile, line);
  stream << line;
  for (int i = 0; i < M; ++i)
    stream >> alpha(i);
  stream.str("");
  stream.clear();

  T.resize(M);
  for (int i = 0; i < M; ++i)
  {
    T[i].load_weight(infile, stream);
    stream.str("");
    stream.clear();
  }
}

Eigen::MatrixXd Adaboost::cal_confusion_matrix(const Eigen::VectorXd &y, const Eigen::VectorXd &pred_Y)
{
  uint32_t R = y.size();

  for (uint32_t i = 0; i < R; ++i)
  {
    if (y(i) == 0 && pred_Y(i) == 0)
      ++TN;
    else if (y(i) == 0 && pred_Y(i) == 1)
      ++FP;
    else if (y(i) == 1 && pred_Y(i) == 1)
      ++TP;
    else if (y(i) == 1 && pred_Y(i) == 0)
      ++FN;
  }

  Eigen::MatrixXd outMatrix(2, 2);
  outMatrix << TP, FP, FN, TN;
  return outMatrix;
}