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

  for (int i = 0; i < M; ++i) {
    w /= w.sum();
    const auto [pred_Y, err] = T[i].fit(train_X, train_Y, w, 1500);

    if (err == 1.0) {
      auto tmp = T[i];
      T.clear();
      T.push_back(std::move(tmp));
      M = 1;
      alpha = Eigen::VectorXd::Ones(M);
      break;
    }

    alpha(i) = std::log((1 + err) / (1 - err)) / 2;
    for (int r = 0; r < train_Y.size(); ++r) {
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

  return C.unaryExpr([](double x) { return double(x > 0); });
}

void Adaboost::store_weight(const std::string filepath)
{
  // compact the correct rate between storing file and current training data
  std::ifstream infile(filepath);
  if (infile.fail()) {
    std::cerr << "cant read " << filepath << '\n';
    exit(1);
  }

  uint32_t store_TN, store_TP, store_FN, store_FP;
  infile >> store_TN >> store_TP >> store_FN >> store_FP;
  infile.close();

  double store_TP_FN = static_cast<double>(store_TP) / (store_TP + store_FN);
  double store_TN_FP = static_cast<double>(store_TN) / (store_TN + store_FP);
  double current_TP_FN = static_cast<double>(TP) / (TP + FN);
  double current_TN_FP = static_cast<double>(TN) / (TN + FP);
  if (current_TP_FN <= store_TP_FN) {    // compact TP / TP + FN
    if (current_TP_FN == store_TP_FN) {    // if the two are same, compact TN / TN + FP
      if (current_TN_FP <= store_TN_FP) {
        std::cout << "This weight won't be saved since its correct rate is lower than the original one\n";
        return;
      }
    }
  }

  // store file
  std::ofstream outfile(filepath);
  if (outfile.fail()) {
    std::cerr << "cant found " << filepath << '\n';
    exit(1);
  }
  else
    std::cerr << "Successfully opened " << filepath << '\n';

  outfile << current_TP_FN << ' ' << current_TN_FP << '\n';

  outfile << M << '\n';
  for (int i = 0; i < M; ++i)
    outfile << alpha(i) << " \n"[i == M - 1];

  for (int i = 0; i < M; ++i)
    T[i].store_weight(outfile);

  outfile.close();
}

void Adaboost::load_weight(const std::string filepath)
{
  std::ifstream infile(filepath);
  if (infile.fail()) {
    std::cout << "cant found " << filepath << '\n';
    exit(1);
  }

  std::string line;
  std::stringstream stream;

  getline(infile, line);
  stream << line;
  stream >> TN >> TP >> FN >> FP;
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
  for (int i = 0; i < M; ++i) {
    T[i].load_weight(infile, stream);
    stream.str("");
    stream.clear();
  }

  infile.close();
}

Eigen::MatrixXd Adaboost::cal_confusion_matrix(const Eigen::VectorXd &y, const Eigen::VectorXd &pred_Y)
{
  uint32_t R = y.size();

  for (uint32_t i = 0; i < R; ++i) {
    if (y(i) == 0 && pred_Y(i) == 0)
      ++TN;
    else if (y(i) == 0 && pred_Y(i) == 1)
      ++FP;
    else if (y(i) == 1 && pred_Y(i) == 1)
      ++TP;
    else if (y(i) == 1 && pred_Y(i) == 0)
      ++FN;
  }

  Eigen::MatrixXd out(2, 2);
  out << TP, FP, FN, TN;
  return out;
}