#pragma once
#ifndef __SEGMENT
#define __SEGMENT

#include <tuple>
#include <Eigen/Eigen>

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, uint32_t> Segment(const Eigen::MatrixXd &data)
{
  const auto &x = data.col(0);
  const auto &y = data.col(1);
  const double threshold = 0.1;

  uint32_t Si = 1, Sn = 1; // The point number in the n segment.
  std::vector<int> n0ind;
  int n0;
  for (int i = 0; i < x.size(); ++i)
  {
    if ((x(i) != 0 || y(i) != 0) && (std::isnormal(x(i)) && std::isnormal(y(i))))
      n0ind.emplace_back(i);
  }
  n0 = n0ind[0];

  int max_Si = 0;
  int ROW = 0, COL = 0;
  for (int i = 1; i < n0; ++i)
  {
    if (std::sqrt(std::pow(x(n0ind[i]) - x(n0ind[i - 1]), 2) + std::pow(y(n0ind[i]) - y(n0ind[i - 1]), 2)) < threshold)
    {
      ++ROW;
    }
    else
    {
      ++COL;
      if (ROW > max_Si) // find the maximum num of segment point
        max_Si = ROW;
    }
  }

  Eigen::MatrixXd Seg(max_Si, COL);
  for (int i = 1; i < n0; ++i)
  {
    if (std::sqrt(std::pow(x(n0ind[i]) - x(n0ind[i - 1]), 2) + std::pow(y(n0ind[i]) - y(n0ind[i - 1]), 2)) < threshold)
    {
      ++Si;
      Seg(Si, Sn) = n0in(i);
    }
    else
    {
      ++Sn;
      Si = 1;
      Seg(Si, Sn) = n0in(i);
    }
  }

  Eigen::VectorXd Si_n = Eigen::VectorXd::Zero(Sn);
  for (int j = 0; j < Sn; ++j)
  {
    int k = 0;
    for (int i = 0; i < Seg.cols(); ++i)
    {
      const auto &ci = Seg.cols(i);
      if (cj(i) != 0)
        ++k;
    }

    Si_n(j) = k;
  }

  return { Seg, Si_n, S_n }
}

#endif