/**
 * @file Controller.cpp
 * @author Mes (mes900903@gmail.com) (Discord: Mes#0903)
 * @brief The base of window controller
 * @version 0.1
 * @date 2023-01-24
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef CONTROLLER_H__
#define CONTROLLER_H__

#include <chrono>
#include <Eigen/Eigen>
#include <string>
#include <fstream>
#include <vector>

class AnimationController {
public:
  void transform_frame();
  void read_frame();

  virtual void check_auto_play();
  virtual void check_update_frame() = 0;

  AnimationController();

public:
  int fps;
  int HZ;
  int frame, max_frame;
  int window_size;

  bool update_frame;
  std::chrono::system_clock::time_point current_time;

  bool auto_play;
  bool replay;

  Eigen::MatrixXd feature_matrix;
  std::vector<Eigen::MatrixXd> segment_vec;

  Eigen::MatrixXd xy_data;
  std::string raw_data_path;
  std::string raw_bin_path;

  std::ifstream raw_bin_file;
  bool raw_bin_open;
  bool is_xydata;
};

#endif