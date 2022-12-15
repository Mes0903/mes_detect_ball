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
  Eigen::MatrixXd cal_confusion_matrix(const Eigen::VectorXd &y, const Eigen::VectorXd &pred_Y);

  /**
   * @brief Transforming the matrix from [theta, r] data to [x, y] data.
   *
   * @param data The [thera, r] matrix.
   * @param ROWS The rows number of the matrix.
   */
  void transform_to_xy(Eigen::MatrixXd &data, const int ROWS);
}    // namespace metric

#endif