/**
 * @file adaboost_classifier.cpp
 * @author Mes (mes900903@gmail.com) (Discord: Mes#0903)
 * @brief The implementation of the Adaboost.
 * @version 0.1
 * @date 2022-11-17
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
  double recall = static_cast<double>(TP) / (TP + FN);
  double precision = static_cast<double>(TP) / (TP + FP);
  double F1_Score = 2 * precision * recall / (precision + recall);

  outfile << F1_Score << ' ' << TN << ' ' << TP << ' ' << FN << ' ' << FP << '\n';
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
  double F1_Score;

  getline(infile, line);
  stream << line;
  stream >> F1_Score >> TN >> TP >> FN >> FP;
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
 * @brief Set the confusion matrix of the Adaboost.
 *
 * @param confusion_matrix The confusion matrix.
 */
void Adaboost::set_confusion_matrix(const Eigen::MatrixXd &confusion_matrix)
{
  TP = static_cast<uint32_t>(confusion_matrix(0, 0));
  FP = static_cast<uint32_t>(confusion_matrix(0, 1));
  FN = static_cast<uint32_t>(confusion_matrix(1, 0));
  TN = static_cast<uint32_t>(confusion_matrix(1, 1));
}

/**
 * @brief Print the confusion table.
 */
void Adaboost::print_confusion_matrix()
{
  Eigen::MatrixXd confusion_matrix(2, 2);
  confusion_matrix << TP, FP, FN, TN;

  std::cout << "The confusion matrix of the Adaboost is : \n"
            << confusion_matrix << '\n';
}