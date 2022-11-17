#pragma once
#ifndef __ADABOOST_CLASSIFIER
#define __ADABOOST_CLASSIFIER

/**
 * @file adaboost_classifier.h
 * @author Mes
 * @brief The declaration of Adaboost class
 * @version 0.1
 * @date 2022-11-17
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "weak_learner.h"
#include "normalize.h"
#include <vector>
#include <Eigen/Eigen>

/**
 * @brief The Adaboost class, have M weak_learner, each weak_learner is a logistic regression classifier.
 */
class Adaboost
{
public:
  uint32_t TN{}, TP{}, FN{}, FP{};
  int M = 0;                   // the number of weak_learner
  Eigen::VectorXd alpha;       // the vector of weights for weak_learner
  std::vector<weak_learner> T; // the vector of weak_learner

public:
  Adaboost() = default;
  Adaboost(const int M)
      : M{M}, T(M) { alpha = Eigen::VectorXd::Zero(M); }

  void fit(const Eigen::MatrixXd &train_X, const Eigen::VectorXd &train_Y); // training Adaboost
  Eigen::VectorXd predict(const Eigen::MatrixXd &test_X);                   // make prediction

public:
  void store_weight(const std::string filepath, std::ofstream &outfile); // store the weights of Adaboost
  void load_weight(const std::string filepath, std::ifstream &infile);   // load the weights before stored.
  void set_confusion_matrix(const Eigen::MatrixXd &confusion_matrix);    // set the confusion matrix of the Adaboost
  void print_confusion_matrix();                                         // print the confusion matrix of the Adaboost
};

#endif