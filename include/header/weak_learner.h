#pragma once
#ifndef WEAK_LEARNER__
#define WEAK_LEARNER__

/**
 * @file weak_learner.h
 * @author Mes (mes900903@gmail.com) (Discord: Mes#0903)
 * @brief The weak learner in Adaboost.
 * @version 0.1
 * @date 2022-11-17
 */

#include <vector>
#include <Eigen/Eigen>
#include <fstream>
#include <tuple>
#include <limits>

/**
 * @brief The weak learner in Adaboost.
 */
class weak_learner
{
public:
  double choose_value;         // w0 in the weight vector
  uint32_t choose_idx;
  double left_label;
  double right_label;

public:
  std::tuple<Eigen::VectorXd, double, bool> fit(const Eigen::MatrixXd &train_X, const Eigen::VectorXd &train_Y, const Eigen::MatrixXd &train_weight, int i); // training
  Eigen::VectorXd predict(const Eigen::MatrixXd &section);   // predict the section data
  Eigen::VectorXd get_label(const Eigen::MatrixXd &section);

public:
  void store_weight(std::ofstream &outfile);                          // store the weight vector
  void load_weight(std::ifstream &infile, std::stringstream &stream); // load the weight vector
};

#endif