#pragma once
#ifndef __ADABOOST_CLASSIFIER
#define __ADABOOST_CLASSIFIER

#include "weak_learner.h"
#include "normalize.h"
#include <vector>
#include <Eigen/Eigen>

class Adaboost {
public:
  uint32_t TN{}, TP{}, FN{}, FP{};
  int M = 0;
  Eigen::VectorXd alpha;
  std::vector<weak_learner> T;

public:
  Adaboost() = default;
  Adaboost(const int M)
      : M{ M }, T(M) { alpha = Eigen::VectorXd::Zero(M); }

  void fit(const Eigen::MatrixXd &train_X, const Eigen::VectorXd &train_Y);
  Eigen::VectorXd predict(const Eigen::MatrixXd &test_X);

public:
  void store_weight(const std::string filepath);
  void load_weight(const std::string filepath);
  Eigen::MatrixXd cal_confusion_matrix(const Eigen::VectorXd &y, const Eigen::VectorXd &pred_Y);
};

#endif