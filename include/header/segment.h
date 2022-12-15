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
std::vector<Eigen::MatrixXd> section_to_segment(const Eigen::MatrixXd &section);    // section is 720*2

#endif
