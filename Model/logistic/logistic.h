#ifndef WEAK_LEARNER__
#define WEAK_LEARNER__

/**
 * @file logistic.h
 * @author Mes (mes900903@gmail.com) (Discord: Mes#0903)
 * @brief The weak learner in Adaboost.
 * @version 0.1
 * @date 2022-11-17
 */

#include <vector>
#include <Eigen/Eigen>
#include <fstream>
#include <tuple>

/**
 * @brief The weak learner in Adaboost.
 */
class logistic {
public:
  double w0;    // w0 in the weight vector
  Eigen::VectorXd w;    // the weight vector

public:
  std::tuple<Eigen::VectorXd, double, bool> fit(const Eigen::MatrixXd &train_X, const Eigen::VectorXd &train_Y, const Eigen::MatrixXd &train_weight, uint32_t Iterations);    // training

  Eigen::ArrayXd cal_logistic(const Eigen::ArrayXd &x);    // logistic function
  Eigen::VectorXd get_label(const Eigen::MatrixXd &section);    // get the label of the section
  Eigen::VectorXd predict(const Eigen::MatrixXd &section);    // predict the section data

public:
  void store_weight(std::ofstream &outfile);    // store the weight vector
  void load_weight(std::ifstream &infile, std::stringstream &stream);    // load the weight vector
};

#endif