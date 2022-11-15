#pragma once
#ifndef __ADABOOST_CLASSIFIER
#define __ADABOOST_CLASSIFIER

#include "weak_learner.h"
#include "normalize.h"
#include <vector>
#include <Eigen/Eigen>

class Adaboost
{
public:
  double correct_rate = 0.0;
  int M = 0;
  Eigen::VectorXd alpha;
  std::vector<weak_learner> T;

public:
  Adaboost() = default;
  Adaboost(const int M) : M{M}, T(M) { alpha = Eigen::VectorXd::Zero(M); }

  void fit(const Eigen::MatrixXd &train_X, const Eigen::VectorXd &train_Y);
  Eigen::VectorXd predict(const Eigen::MatrixXd &test_X);
  void set_classifier_num(const int num);

public:
  void store_weight(const char *filename, uint32_t TP, uint32_t FN, Normalizer &normalizer);
  void load_weight(const char *filename, Normalizer &normalizer);
};

#endif