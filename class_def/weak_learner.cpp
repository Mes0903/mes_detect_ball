/**
 * @file weak_learner.cpp
 * @author Mes (mes900903@gmail.com) (Discord: Mes#0903)
 * @brief The implementation of the weak learner class in Adaboost, I use the logistic as the weak learner.
 * @version 0.1
 * @date 2022-11-17
 */

#include "weak_learner.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <tuple>
#include <random>
#include <fstream>
#include <sstream>
#include <Eigen/Eigen>
#include <functional>
#include <limits>

/**
 * @brief Training the weight in weak learner
 *
 * @param train_X The training data, which is a feature matrix.
 * @param train_Y The training label.
 * @param train_weight The training weight in adaboost.
 * @param Iterations The training iterations.
 * @return std::tuple<Eigen::VectorXd, double, bool> The first element of the pair is the label it predict,
 *                                            the second one is the error rate,
 *                                            the third one is a flag for 100% accuracy, if the accuracy is 100%, we can delete all the other weak learner in adaboost.
 */


double gss(const std::function<double(double)> &f, double a, double b, double tol = 1e-10) {
    double gr = 1.618033988749895, c, d;
    c = b - (b - a) / gr;
    d = a + (b - a) / gr;
    while (abs(b - a) > tol) {
        if (f(c) < f(d)) {
            b = d;
        } else {
            a = c;
        }
        c = b - (b - a) / gr;
        d = a + (b - a) / gr;
    }
    return (b + a) / 2;
};

std::tuple<Eigen::VectorXd, double, bool> weak_learner::fit(const Eigen::MatrixXd &train_X, const Eigen::VectorXd &train_Y, const Eigen::MatrixXd &train_weight, int i)
{
    double log_likelihood = -std::numeric_limits<double>::infinity();
    uint32_t D = train_X.cols(); // dimension is the column of training data, which is 5 in my case, since there is 5 features.
    Eigen::ArrayXd train_Y_array=train_Y, train_weight_array=train_weight;
    for (uint32_t dim = 0; dim < D; ++dim) {
        Eigen::VectorXd choose_data = train_X.col(dim);
        auto func = [&choose_data, &train_weight_array, &train_Y_array](double value) {
            Eigen::ArrayXd left = choose_data.unaryExpr([value](double x) { return double(x < value); });
            Eigen::ArrayXd right = (1 - left);
            Eigen::ArrayXd left_weight = left * train_weight_array, right_weight = right * train_weight_array;
            double left_weight_sum = left_weight.sum();
            double left_weight_1 = (left_weight * train_Y_array).sum();
            double right_weight_sum = right_weight.sum();
            double right_weight_1 = (right_weight * train_Y_array).sum();
            double left_prob = 0, right_prob = 0;
            if (left_weight_sum != 0)
                left_prob = left_weight_1 / left_weight_sum;
            if (right_weight_sum != 0)
                right_prob = right_weight_1 / right_weight_sum;
            Eigen::ArrayXd pred_y = (left * left_prob + right * right_prob) * (1 - 2e-8) + 1e-8;
            return -((train_Y_array * pred_y.log() + (1-train_Y_array) * (1-pred_y).log())*train_weight_array).sum();
        };
        double tmp_value = gss(func, choose_data.minCoeff(),
                               choose_data.maxCoeff()), tmp_left_prob = 0, tmp_right_prob = 0;
        Eigen::ArrayXd left = choose_data.unaryExpr([&tmp_value](double x) { return double(x < tmp_value); });
        Eigen::ArrayXd right = (1 - left);
        Eigen::ArrayXd left_weight = left * train_weight_array, right_weight = right * train_weight_array;
        double left_weight_sum = left_weight.sum();
        double left_weight_1 = (left_weight * train_Y_array).sum();
        double right_weight_sum = right_weight.sum();
        double right_weight_1 = (right_weight * train_Y_array).sum();
        if (left_weight_sum != 0)
            tmp_left_prob = left_weight_1 / left_weight_sum;
        if (right_weight_sum != 0)
            tmp_right_prob = right_weight_1 / right_weight_sum;
        Eigen::ArrayXd pred_y = (left * tmp_left_prob + right * tmp_right_prob) * (1 - 2e-8) + 1e-8;
        double tmp_log_likelihood = ((train_Y_array * pred_y.log() + (1-train_Y_array) * (1-pred_y).log())*train_weight_array).sum();
        if (tmp_log_likelihood > log_likelihood) {
            choose_idx = dim;
            choose_value = tmp_value;
            left_label = tmp_left_prob >= 0.5;
            right_label = tmp_right_prob >= 0.5;
            log_likelihood = tmp_log_likelihood;
        }
    }

  Eigen::VectorXd pred_Y = get_label(train_X);

  double err = 0.0;
  bool all_correct = true;
  for (int i = 0; i < pred_Y.size(); ++i)
  {
    if (pred_Y(i) != train_Y(i))
    {
      all_correct = false;
      err -= train_weight(i);
    }
    else
    {
      err += train_weight(i);
    }
  }

  return {pred_Y, err, all_correct};
}
/**
 * @brief Make prediction of the data, this function is for debug using, it's output is not an label but an probability.
 *
 * @param data The feature matrix of all section, the size is Sn*5, Sn is the total number of the data, 5 means the number of the feature.
 * @return Eigen::VectorXd The probability of the data get from the logistic function.
 */
Eigen::VectorXd weak_learner::predict(const Eigen::MatrixXd &data)
{
    Eigen::ArrayXd out = data.col(choose_idx).unaryExpr([this](double x)
                                      { return double(x < choose_value); });
  return out * left_label + (1 - out) * right_label;
}

Eigen::VectorXd weak_learner::get_label(const Eigen::MatrixXd &data)
{
    Eigen::ArrayXd out = data.col(choose_idx).unaryExpr([this](double x)
                                      { return double(x < choose_value); });
  return out * left_label + (1 - out) * right_label;
}

/**
 * @brief Predict the label of the data.
 *
 * @param data The feature matrix of all section, the size is Sn*5, Sn is the total number of the data, 5 means the number of the feature.
 * @return Eigen::VectorXd The label of the data. If the probability get from the logistic function >= 0.5, output 1, otherwise 0.
 */

/**
 * @brief Store the weight of the weak learner.
 *
 * @param outfile The file path, where to store the weight.
 */
void weak_learner::store_weight(std::ofstream &outfile)
{
  outfile << choose_value << ' ';
  outfile << choose_idx << ' ';
  outfile << left_label << ' ';
  outfile << right_label << '\n';
}

/**
 * @brief Load the weight of the weak learner.
 *
 * @param infile The file path, where to load the weight.
 * @param stream For avoiding copying std::stringstream, pass it by reference into function.
 */
void weak_learner::load_weight(std::ifstream &infile, std::stringstream &stream)
{
  std::string line;
  getline(infile, line);
  stream << line;
  stream >> choose_value;
  stream >> choose_idx;
  stream >> left_label;
  stream >> right_label;
}