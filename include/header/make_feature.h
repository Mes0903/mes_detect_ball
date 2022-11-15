#pragma once
#ifndef __MAKE_FEATURES
#define __MAKE_FEATURES

#include "segment.h"

#include <vector>
#include <utility>
#include <cmath>
#include <Eigen/Eigen>

Eigen::MatrixXd AppendRow(const Eigen::MatrixXd &A, const Eigen::VectorXd &B)
{
  Eigen::MatrixXd D(A.rows() + 1, A.cols());
  D << A, B.transpose();
  return D;
}

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

std::pair<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>> transform_to_feature(const Eigen::MatrixXd &xy_data)
{
  std::vector<Eigen::MatrixXd> segment_data;
  Eigen::MatrixXd feature_data = Eigen::MatrixXd::Zero(1, 5);
  bool empty_flag = true;

  int SECTION = xy_data.rows() / 720;

  for (int i = 0; i < 60; ++i)
  {
    int n = 1 + 15 * i;
    int begin_index = 720 * (n - 1);
    Eigen::MatrixXd section_data = xy_data.block(begin_index, 0, 720, 2);            // section_data is 720*2
    std::vector<Eigen::MatrixXd> section_seg_vec = do_section_segment(section_data); // 那一秒切出來的所有 segment

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
    segment_data.insert(segment_data.end(),
                        std::make_move_iterator(section_seg_vec.begin()),
                        std::make_move_iterator(section_seg_vec.end()));
  }

  return {feature_data, segment_data};
}

#endif