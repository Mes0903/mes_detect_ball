#pragma once
#ifndef SEGMENT__
#define SEGMENT__

/**
 * @file segment.h
 * @author Mes (mes900903@gmail.com) (Discord: Mes#0903)
 * @brief Classify the xy data into segments data. I JUST USE THIS FILE FOR CLASSFIED ROBOT DETECTION MATRIX, TRAINING DATA AND TEST DATA I USED MATLAB TO CLASSIFIED SEGMENT.
 * @version 0.1
 * @date 2022-11-17
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
std::vector<Eigen::MatrixXd> section_to_segment(const Eigen::MatrixXd &section) // section is 720*2
{
  const auto &x = section.col(0);
  const auto &y = section.col(1);
  const double threshold = 0.1;
  std::vector<Eigen::MatrixXd> seg_vec; // a seg is a n*2 matrix.

  std::vector<int> valid_index; // valid point index
  for (int i = 0; i < 720; ++i)
  {
    if ((x(i) != 0 || y(i) != 0) && (std::isnormal(x(i)) && std::isnormal(y(i)))) // if the xy is [0,0], it's not valid; if the x or y is sth like nan, it's not valid too
      valid_index.emplace_back(i);
  }
  int validsize = valid_index.size(); // the number of valid point in the section.
  bool first_end = std::sqrt(std::pow(x(valid_index[0]) - x(valid_index[validsize - 1]), 2) + std::pow(y(valid_index[0]) - y(valid_index[validsize - 1]), 2)) < threshold;

  std::vector<int> single_seg; // The valid xy point index list of one segment.

  single_seg.emplace_back(valid_index[0]); // The first point
  for (int i = 1; i < validsize; ++i)      // traverse all valid point and devide it into segment
  {
    // if std::sqrt( (x1 - x0)^2 + (y1-y0)^2 ) >= threshold, it's means the previous points in the `single_seq` are belongs to same segment.
    // Thus make it as a matrix and push it into the seg_vec
    if (std::sqrt(std::pow(x(valid_index[i - 1]) - x(valid_index[i]), 2) + std::pow(y(valid_index[i - 1]) - y(valid_index[i]), 2)) >= threshold)
    {
      const int node_num = single_seg.size(); // the numbers of the point in the one segment
      Eigen::MatrixXd tmp_seg(node_num, 2);
      for (int j = 0; j < node_num; ++j)
      {
        tmp_seg(j, 0) = x(single_seg[j]);
        tmp_seg(j, 1) = y(single_seg[j]);
      }

      seg_vec.push_back(std::move(tmp_seg)); // push the matrix into the segment list, which represents the xy data of a segment
      single_seg.clear();
    }

    single_seg.emplace_back(valid_index[i]); // The next segment start point.
  }

  const int node_num = single_seg.size(); // the numbers of the point in the one segment
  Eigen::MatrixXd tmp_seg(node_num, 2);
  for (int j = 0; j < node_num; ++j)
  {
    tmp_seg(j, 0) = x(single_seg[j]);
    tmp_seg(j, 1) = y(single_seg[j]);
  }

  seg_vec.push_back(std::move(tmp_seg)); // push the matrix into the segment list, which represents the xy data of a segment
  single_seg.clear();

  if (first_end)
  {
    Eigen::MatrixXd &first = seg_vec[0];
    Eigen::MatrixXd &end = seg_vec[seg_vec.size() - 1];

    Eigen::MatrixXd tmp(first.rows() + end.rows(), first.cols());
    tmp << first, end;

    seg_vec[0] = std::move(tmp);
    seg_vec.pop_back();
  }

  return seg_vec;
}

#endif