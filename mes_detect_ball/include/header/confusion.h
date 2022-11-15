#pragma once
#ifndef __CONFUSION
#define __CONFUSION

#include <Eigen/Eigen>
#include <iostream>

Eigen::MatrixXd cal_confusion_matrix(const Eigen::VectorXd &y, const Eigen::VectorXd &pred_Y)
{
  uint32_t R = y.size();
  uint32_t TN{}, TP{}, FN{}, FP{};

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

  Eigen::MatrixXd out(2, 2);
  out << TP, FP, FN, TN;
  return out;
}

#endif