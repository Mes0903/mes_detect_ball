#pragma once
#ifndef __SEGMENT
#define __SEGMENT

#include <tuple>
#include <vector>
#include <Eigen/Eigen>

std::vector<Eigen::MatrixXd> do_section_segment(const Eigen::MatrixXd &section) // section is 720*2
{
  const auto &x = section.col(0);
  const auto &y = section.col(1);
  const double threshold = 0.1;
  std::vector<Eigen::MatrixXd> seg_vec; // a seg is a n*2 matrix.

  std::vector<int> valid_index; // valid point index
  for (int i = 0; i < 720; ++i)
  {
    if ((x(i) != 0 || y(i) != 0) && (std::isnormal(x(i)) && std::isnormal(y(i))))
      valid_index.emplace_back(i);
  }
  int validsize = valid_index.size();

  std::vector<int> seg_index_list; // 某個 Seg 的 index list

  for (int i = 1; i < validsize; ++i) // 遍歷所有 valid index
  {
    if (std::sqrt(std::pow(x(valid_index[i - 1]) - x(valid_index[i]), 2) + std::pow(y(valid_index[i - 1]) - y(valid_index[i]), 2)) < threshold)
    {
      // 將 valid index 插入 seg list 內
      seg_index_list.emplace_back(valid_index[i - 1]);
    }
    else
    {
      int node_num = seg_index_list.size(); // 這個 segment 總共有幾個點
      if (node_num != 0)
      {
        Eigen::MatrixXd tmp_seg(node_num, 2);
        for (int j = 0; j < node_num; ++j)
        {
          tmp_seg(j, 0) = x(seg_index_list[j]);
          tmp_seg(j, 1) = y(seg_index_list[j]);
        }

        seg_vec.push_back(std::move(tmp_seg));
        seg_index_list.clear();
      }
    }
  }

  return seg_vec;
}

#endif