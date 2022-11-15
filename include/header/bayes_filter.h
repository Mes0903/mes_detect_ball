#pragma once
#ifndef __BAYES_FILTER
#define __BAYES_FILTER

#include "segment.h"
#include <Eigen/Eigen>

extern Adaboost A;

int IsSegInBuffer(const Eigen::MatrixXd &buffer, const Eigen::MatrixXd &Seg)
{
  double x_mean = Seg.col(0).sum() / (Seg.rows());
  double y_mean = Seg.col(1).sum() / (Seg.rows());
  int distance = 10; // hyperparameter
  double x, y;
  for (int i = 0; i < buffer.size(); i++)
  {
    x = buffer(i, 0);
    y = buffer(i, 1);
    if (std::sqrt(std::pow((x - x_mean), 2) + std::pow((y - y_mean), 2)) < distance)
      return i;
  }

  return -1;
}

// buffer 存 label 出現 1 的 seg 的質心座標 + 是 1 的機率 [x,y,p]
Eigen::MatrixXd Bayes_filter(const Eigen::MatrixXd &T, const Eigen::MatrixXd &Z,
                             double p1, double p0, const Eigen::MatrixXd &xy_data, Eigen::MatrixXd &buffer)
{
  std::vector<Eigen::MatrixXd> Seg = do_section_segment(xy_data);
  int S_n = Seg.size();

  Eigen::VectorXd V(2);
  Eigen::VectorXd U;
  int index;
  double p, eta;
  for (int i = 0; i < S_n; i++)
  {
    index = IsSegInBuffer(buffer, Seg(i));
    if (A.predict(Seg[i]) == 1)
    {
      if (index != -1)
      {
        // iteration of Bayes filter
        p = buffer(index, 2);
        V << p, 1 - p;
        U = T.transpose() * V;
        V = Z.col(0).cwiseProduct(U);
        eta = 1 / V.sum();
        V = V * eta;

        buffer(index, 0) = Seg[i].col(0).sum() / (Seg[i].rows()); // Seg(S_n)的質心的x座標
        buffer(index, 1) = Seg[i].col(1).sum() / (Seg[i].rows()); // Seg(S_n)的質心的y座標
        buffer(index, 2) = V(0);
      }
      else
      {
        p = Z(0, 0) * p1 / (Z(0, 0) * p1 + Z(1, 0) * p0); // first probability P( x0 | Z0 )
                                                          // code: 在buffer下加一row存 [Seg(S_n)的x座標, Seg(S_n)的y座標, p]
      }
    }
    else
    {
      if (index != -1)
      {
        // iteration of Bayes filter
        p = buffer(index, 2);
        V << p, 1 - p;
        U = T.transpose() * V;
        V = Z.col(1).cwiseProduct(U);
        eta = 1 / V.sum();
        V = V * eta;

        buffer(index, 0) = Seg[i].col(0).sum() / (Seg[i].rows());
        buffer(index, 1) = Seg[i].col(1).sum() / (Seg[i].rows());
        buffer(index, 2) = V(0);
      }
    }
  }
  return buffer
}

#endif