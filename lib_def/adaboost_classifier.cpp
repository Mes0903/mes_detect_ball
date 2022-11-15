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

void Adaboost::set_classifier_num(const int num)
{
  M = num;
  T.resize(num);
  alpha = Eigen::VectorXd::Zero(M);
}

void Adaboost::store_weight(const char *filename, uint32_t TP, uint32_t FN, Normalizer &normalizer)
{
  std::ifstream infile(filename);
  if (infile.fail())
  {
    std::cerr << "cant read " << filename << '\n';
    exit(1);
  }

  double correct_rate;
  infile >> correct_rate;
  infile.close();
  if (static_cast<double>(TP) / (TP + FN) < correct_rate)
  {
    std::cout << "This weight won't be saved since its correct rate is lower than the original one\n";
    return;
  }

  std::ofstream outfile(filename);
  if (outfile.fail())
  {
    std::cerr << "cant found " << filename << '\n';
    exit(1);
  }
  else
    std::cerr << "Successfully opened " << filename << '\n';

  outfile << static_cast<double>(TP) / (TP + FN) << '\n';
  outfile << normalizer.data_min.size() << ' ' << normalizer.data_mm.size() << '\n';
  for (int i = 0; i < normalizer.data_min.size(); ++i)
    outfile << normalizer.data_min(i) << " \n"[i == normalizer.data_min.size() - 1];

  for (int i = 0; i < normalizer.data_mm.size(); ++i)
    outfile << normalizer.data_mm(i) << " \n"[i == normalizer.data_mm.size() - 1];

  outfile << M << '\n';
  for (int i = 0; i < M; ++i)
    outfile << alpha(i) << " \n"[i == M - 1];

  for (int i = 0; i < M; ++i)
    T[i].store_weight(outfile);

  outfile.close();
  std::cerr << "Successfully store " << filename << '\n';
}

void Adaboost::load_weight(const char *filename, Normalizer &normalizer)
{
  std::ifstream infile(filename);
  if (infile.fail())
  {
    std::cout << "cant found " << filename << '\n';
    exit(1);
  }

  std::string line;
  std::stringstream stream;

  getline(infile, line);
  stream << line;
  stream >> correct_rate;
  stream.str("");
  stream.clear();

  int min_size, mm_size;
  getline(infile, line);
  stream << line;
  stream >> min_size >> mm_size;
  stream.str("");
  stream.clear();
  normalizer.data_min = Eigen::VectorXd::Zero(min_size);
  normalizer.data_mm = Eigen::VectorXd::Zero(mm_size);

  getline(infile, line);
  stream << line;
  for (int i = 0; i < min_size; ++i)
    stream >> normalizer.data_min(i);

  stream.str("");
  stream.clear();

  getline(infile, line);
  stream << line;
  for (int i = 0; i < mm_size; ++i)
    stream >> normalizer.data_mm(i);

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

  infile.close();
}