#pragma once
#ifndef __MAKE_FEATURES
#define __MAKE_FEATURES

/**
 * @file make_feature.h
 * @author Mes (mes900903@gmail.com) (Discord: Mes#0903)
 * @brief Calculate the feature of the matrix, I used it to transform the laser data from segment to feature matrix.
 * @version 0.1
 * @date 2022-11-17
 */

#include "segment.h"

#include <vector>
#include <utility>
#include <cmath>
#include <Eigen/Eigen>

/**
 * @brief Append a row to the matrix.
 *
 * @param A The target matrix wanna append a row.
 * @param B The vector would be appended.
 * @return Eigen::MatrixXd The result matrix after complete appending.
 */
Eigen::MatrixXd AppendRow(const Eigen::MatrixXd &A, const Eigen::VectorXd &B)
{
  Eigen::MatrixXd D(A.rows() + 1, A.cols());
  D << A, B.transpose();
  return D;
}

/**
 * @brief Calculate the point of the segment.
 *
 * @param Seg The segment data matrix. It's an Sn*2 matrix, Sn is the number of segments.
 * @return uint32_t The points number of the segment, which is the Sn above.
 */
uint32_t cal_point(const Eigen::MatrixXd &Seg)
{
  return Seg.rows();
}

/**
 * @brief Calculate the standard deviation of the segment.
 *
 * @param Seg The segment data matrix. It's an Sn*2 matrix, Sn is the number of segments.
 * @return double The standard deviation.
 */
double cal_std(const Eigen::MatrixXd &Seg)
{
  uint32_t n = cal_point(Seg);
  if (n < 2)
    return 0.0;

  Eigen::Vector2d m = Seg.colwise().mean(); // the means vector of data
  const auto &x = Seg.col(0);
  const auto &y = Seg.col(1);

  // ( (1/n-1) * ( sum( (x-x_mean)^2 )+ sum( (y-y_mean)^2 ) ) ^ (1/2)
  double sigma = std::sqrt((1 / (n - 1)) * ((x.array() - m(0)).square().sum() + (y.array() - m(1)).square().sum()));

  return sigma;
}

/**
 * @brief Calculate the width of the segment, which is the distance of first point and the last point.
 *
 * @param Seg The segment data matrix. It's an Sn*2 matrix, Sn is the number of segments.
 * @return double The width
 */
double cal_width(const Eigen::MatrixXd &Seg)
{
  // ( (x0 - x_last)^2 + (y0 - y_last)^2 )^(1/2)
  double width = std::sqrt(std::pow(Seg(0, 0) - Seg(Seg.rows() - 1, 0), 2) + std::pow(Seg(0, 1) - Seg(Seg.rows() - 1, 1), 2));
  return width;
}

/**
 * @brief Calculate the circularity and the radius of the segment.
 *
 * @param Seg The segment data matrix. It's an Sn*2 matrix, Sn is the number of segments.
 * @return std::pair<double, double>
 */
std::pair<double, double> cal_cr(const Eigen::MatrixXd &Seg)
{
  const auto &x = Seg.col(0);
  const auto &y = Seg.col(1);

  Eigen::MatrixXd A(Seg.rows(), 3);
  Eigen::MatrixXd b(Seg.rows(), 1);

  A << -2 * x, -2 * y, Eigen::MatrixXd::Ones(Seg.rows(), 1);
  b << (-1 * x.array().square() - y.array().square());

  Eigen::MatrixXd x_p = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
  auto xc = x_p(0);
  auto yc = x_p(1);

  double radius = std::sqrt(std::pow(xc, 2) + std::pow(yc, 2) - x_p(2));
  auto circularity = ((radius - ((xc - x.array()).square() + (yc - y.array()).square()).sqrt()).square()).sum();

  return {radius, circularity};
}

/**
 * @brief Making the feature matrix by the segment data.
 *
 * @param Seg The segment data matrix. It's an Sn*2 matrix, Sn is the number of segments.
 * @return Eigen::VectorXd
 */
Eigen::VectorXd make_feature(const Eigen::MatrixXd &Seg)
{
  Eigen::VectorXd feature = Eigen::VectorXd::Zero(5);
  feature(0) = cal_point(Seg);
  feature(1) = cal_std(Seg);
  feature(2) = cal_width(Seg);
  const auto [r, cir] = cal_cr(Seg);
  feature(3) = r, feature(4) = cir;

  return feature;
}

/**
 * @brief Transform the section xy data to the feature matrix.
 *
 * @param xy_data The section xy data. On my minibots, the matrix is 720*2
 * @return std::pair<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>> feature matrix and a vector containing all the segments in one second.
 */
std::pair<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>> section_to_feature(const Eigen::MatrixXd &xy_data) // xy_data is 720*2
{
  Eigen::MatrixXd feature_data = Eigen::MatrixXd::Zero(1, 5);
  bool empty_flag = true;

  std::vector<Eigen::MatrixXd> section_seg_vec = section_to_segment(xy_data); // 那一秒切出來的所有 segment

  for (const auto &Seg : section_seg_vec)
  {
    Eigen::VectorXd single_feature = make_feature(Seg);

    if (empty_flag)
    {
      feature_data += single_feature.transpose();
      empty_flag = false;
    }
    else
    {
      feature_data = AppendRow(feature_data, single_feature);
    }
  }

  return {feature_data, section_seg_vec};
}

#endif