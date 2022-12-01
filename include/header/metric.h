#pragma once
#ifndef METRIC_H__
#define METRIC_H__

/**
 * @file metric.h
 * @author Mes (mes900903@gmail.com) (Discord: Mes#0903)
 * @brief The metric related functions.
 * @version 0.1
 * @date 2022-11-18
 */

#include <tuple>
#include <cmath>
#include <Eigen/Eigen>

namespace metric {
  /**
   * @brief Calculate the confusion table.
   *
   * @param y The label of the data.
   * @param pred_Y The predicted output of the data.
   * @return Eigen::MatrixXd The confusion table.
   */
  Eigen::MatrixXd cal_confusion_matrix(const Eigen::VectorXd &y, const Eigen::VectorXd &pred_Y)
  {
    int R = y.size();
    int TP{}, FP{}, FN{}, TN{};

    for (int i = 0; i < R; ++i) {
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

  /**
   * @brief Transforming the matrix from [theta, r] data to [x, y] data.
   *
   * @param data The [thera, r] matrix.
   * @param ROWS The rows number of the matrix.
   */
  void transform_to_xy(Eigen::MatrixXd &data, const int ROWS)
  {
    for (int i = 0; i < ROWS; i++) {
      const double theta = M_PI * data(i, 0) / 180;    // transform the radian to angle
      const double r = data(i, 1);    // radius

      data(i, 0) = r * std::cos(theta);    // x
      data(i, 1) = r * std::sin(theta);    // y
    }
  }
}    // namespace metric

#endif