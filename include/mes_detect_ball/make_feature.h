#pragma once
#ifndef __MAKE_FEATURES
#define __MAKE_FEATURES

#include <vector>
#include <utility>
#include <cmath>
#include <Eigen/Eigen>

uint32_t cal_point(const Eigen::MatrixXd &data)
{
  return data.rows();
}

double cal_std(const Eigen::MatrixXd &data)
{
  uint32_t n = cal_point(data);
  Eigen::Vector2d m = data.colwise().mean();
  const auto &x = data.col(0);
  const auto &y = data.col(1);
  double sigma = std::sqrt(1 / n * ((x.array() - m(0)).square().sum() + (y.array() - m(1)).square().sum()));

  return sigma;
}

double cal_width(const Eigen::MatrixXd &data)
{
  double width = std::sqrt(std::pow(data(0, 0) - data(data.rows() - 1, 0), 2) + std::pow(data(0, 1) - data(data.rows() - 1, 1), 2));
  return width;
}

std::pair<double, double> cal_cr(const Eigen::MatrixXd &data)
{
  const auto &x = data.col(0);
  const auto &y = data.col(1);

  Eigen::MatrixXd A(data.rows(), 3);
  Eigen::MatrixXd b(data.rows(), 1);

  A << -2 * x, -2 * y, Eigen::MatrixXd::Ones(data.rows(), 1);
  b << (-1 * x.array().square() - y.array().square());

  Eigen::MatrixXd x_p = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
  auto xc = x_p(0);
  auto yc = x_p(1);

  double radius = std::sqrt(std::pow(xc, 2) + std::pow(yc, 2) - x_p(2));
  auto circularity = ((radius - ((xc - x.array()).square() + (yc - y.array()).square()).sqrt()).square()).sum();

  return {radius, circularity};
}

#endif