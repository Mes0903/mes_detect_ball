/**
 * @file adaboost_classifier.cpp
 * @author Mes
 * @brief The implementation of the Adaboost.
 * @version 0.1
 * @date 2022-11-17
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "adaboost_classifier.h"
#include "normalize.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <Eigen/Eigen>

/**
 * @brief Training Adaboost.
 *
 * @param train_X The training data, which is a feature matrix.
 * @param train_Y The training label.
 */
void Adaboost::fit(const Eigen::MatrixXd &train_X, const Eigen::VectorXd &train_Y)
{
  Eigen::VectorXd w = Eigen::VectorXd::Ones(train_X.rows());

  for (int i = 0; i < M; ++i)
  {
    w /= w.sum();
    const auto [pred_Y, err, all_correct] = T[i].fit(train_X, train_Y, w, 1500); // pred_Y is the label it predict, err is the error rate.

    // if the accuracy is 100%, we can delete all the other weak learner in adaboost, just use this weak learner to judge data.
    if (all_correct)
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

/**
 * @brief Make the prediction of the data.
 *
 * @param data The data need to be predicted, which is a feature matrix.
 * @return Eigen::VectorXd The output label vector.
 */
Eigen::VectorXd Adaboost::predict(const Eigen::MatrixXd &data)
{
  uint32_t R = data.rows();
  Eigen::ArrayXd C(R);
  for (int m = 0; m < M; ++m)
    C += alpha(m) * (2 * T[m].get_label(data).array() - 1);

  return C.unaryExpr([](double x)
                     { return double(x > 0); });
}

/**
 * @brief Store the weight vector of all weak learner in Adaboost.
 *
 * @param filepath For Debug using, maybe unused. The file path, where to store the weight.
 * @param outfile The file, where to store the weight, provided by the file handler.
 */
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

/**
 * @brief Store the weight vector of all weak learner in Adaboost.
 *
 * @param filepath For Debug using, maybe unused. The file path, where to load the weight.
 * @param outfile The file, where to load the weight, provided by the file handler.
 */
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

/**
 * @brief Calculate the confusion table.
 *
 * @param y The label of the data.
 * @param pred_Y The predicted output of the data.
 * @return Eigen::MatrixXd The confusion table.
 */
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

  Eigen::MatrixXd confusion(2, 2);
  confusion << TP, FP, FN, TN;
  return confusion;
}