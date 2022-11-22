#pragma once
#ifndef BAYES_FILTER__
#define BAYES_FILTER__

/**
 * @file bayes_filter.h
 * @author Mes (mes900903@gmail.com) (Discord: Mes#0903)
 * @brief calculate the bayes filter
 * @version 0.1
 * @date 2022-11-17
 */

#include "segment.h"
#include <Eigen/Eigen>
#include "make_feature.h"

extern Adaboost A;

/**
 * @brief
 *
 * @param buffer
 * @param Seg
 * @return int
 */
int IsSegInBuffer(const Eigen::MatrixXd &buffer, const Eigen::MatrixXd &Seg)
{
  double x_mean = Seg.col(0).sum() / (Seg.rows());
  double y_mean = Seg.col(1).sum() / (Seg.rows());
  int distance = 0.1; // hyperparameter
  double x, y;
  for (int i = 0; i < buffer.rows(); i++)
  {
    x = buffer(i, 0);
    y = buffer(i, 1);
    if (std::sqrt(std::pow((x - x_mean), 2) + std::pow((y - y_mean), 2)) < distance)
      return i;
  }

  return -1;
}

// buffer 存 label 出現 1 的 seg 的質心座標 + 是 1 的機率 [x,y,p]
/**
 * @brief
 *
 * @param T
 * @param Z
 * @param p1
 * @param p0
 * @param xy_data
 * @param buffer
 * @return Eigen::MatrixXd
 */
Eigen::MatrixXd Bayes_filter(const Eigen::MatrixXd &T, const Eigen::MatrixXd &Z, double p1, double p0, const Eigen::MatrixXd &data, Eigen::MatrixXd &buffer, bool &empty_buffer)
{
  auto [feature_matrix, Seg] = section_to_feature(data);
  Eigen::VectorXd pred_Y = A.predict(feature_matrix); 
  int S_n = Seg.size();

  Eigen::VectorXd V(2);
  Eigen::VectorXd U;
  int index;
  double p, eta;
  
  for (int i = 0; i < S_n; i++)
  {
    index = IsSegInBuffer(buffer, Seg[i]);
    if (pred_Y[i] == 1)
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
        float x = Seg[i].col(0).sum() / (Seg[i].rows());
        float y = Seg[i].col(1).sum() / (Seg[i].rows());
        Eigen::VectorXd temp_vec(3);
        temp_vec << x, y, p;
        if (empty_buffer)
        {
          buffer += temp_vec.transpose();
          empty_buffer = false;
        }
        else {
          buffer = AppendRow(buffer, temp_vec);      // 在buffer下加一row存 [Seg(S_n)的x座標, Seg(S_n)的y座標, p]
        }
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
  return buffer;
}

#endif