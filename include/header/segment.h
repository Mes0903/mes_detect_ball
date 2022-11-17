#pragma once
#ifndef __SEGMENT
#define __SEGMENT

/**
 * @file segment.h
 * @author Mes
 * @brief Classify the xy data into segments data. I JUST USE THIS FILE FOR CLASSFIED ROBOT DETECTION MATRIX, TRAINING DATA AND TEST DATA I USED MATLAB TO CLASSIFIED SEGMENT.
 * @version 0.1
 * @date 2022-11-17
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <vector>
#include <Eigen/Eigen>

/**
 * @brief Transform the xy data to segments data.
 *
 * @param section A section of the xy data, in my case, laser data each seconds is a section, thus the section is a 720*2 matrix.
 * @return std::vector<Eigen::MatrixXd> The std::vector of the segments,
 *         i.e., vec[0] is the first segment, vec[1] is the second segment.
 */
std::vector<Eigen::MatrixXd> do_section_segment(const Eigen::MatrixXd &section)    // section is 720*2
{
  const auto &x = section.col(0);
  const auto &y = section.col(1);
  const double threshold = 0.1;
  std::vector<Eigen::MatrixXd> seg_vec;    // a seg is a n*2 matrix.

  std::vector<int> valid_index;    // valid point index
  for (int i = 0; i < 720; ++i) {
    if ((x(i) != 0 || y(i) != 0) && (std::isnormal(x(i)) && std::isnormal(y(i))))    // if the xy is [0,0], it's not valid; if the x or y is sth like nan, it's not valid too
      valid_index.emplace_back(i);
  }
  int validsize = valid_index.size();

  std::vector<int> seg_index_list;    // The xy data index list of one segment.

  for (int i = 1; i < validsize; ++i)    // traverse all valid point and devide it into segment
  {
    // if std::sqrt( (x1 - x0)^2 + (y1-y0)^2 ) < threshold, it's belongs same segment.
    if (std::sqrt(std::pow(x(valid_index[i - 1]) - x(valid_index[i]), 2) + std::pow(y(valid_index[i - 1]) - y(valid_index[i]), 2)) < threshold) {
      // push the index into the segment list
      seg_index_list.emplace_back(valid_index[i - 1]);
    }
    else {
      int node_num = seg_index_list.size();    // the numbers of the point in the segment
      if (node_num != 0) {
        Eigen::MatrixXd tmp_seg(node_num, 2);
        for (int j = 0; j < node_num; ++j) {
          tmp_seg(j, 0) = x(seg_index_list[j]);
          tmp_seg(j, 1) = y(seg_index_list[j]);
        }

        seg_vec.push_back(std::move(tmp_seg));    // push the matrix into the segment list, which represents the xy data of a segment
        seg_index_list.clear();
      }
    }
  }

  return seg_vec;
}

#endif