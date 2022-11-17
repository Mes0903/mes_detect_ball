#pragma once
#ifndef __METRIC_H__
#define __METRIC_H__

#include <tuple>
#include <Eigen/Eigen>

namespace metric
{
  /**
   * @brief Calculate the confusion table.
   *
   * @param y The label of the data.
   * @param pred_Y The predicted output of the data.
   * @return Eigen::MatrixXd The confusion table.
   */
  Eigen::MatrixXd cal_confusion_matrix(const Eigen::VectorXd &y, const Eigen::VectorXd &pred_Y)
  {
    uint32_t R = y.size();
    uint32_t TP{}, FP{}, FN{}, TN{};

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
}

#endif