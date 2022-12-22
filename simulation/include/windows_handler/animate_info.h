#ifndef ANIMATE_INFORMATION_H__
#define ANIMATE_INFORMATION_H__

#include <chrono>
#include <Eigen/Eigen>

struct AnimationInfo {
  Eigen::MatrixXd xy_data = Eigen::MatrixXd::Zero(720, 2);

  double Ball_X = 0.0, Ball_Y = 0.0, Box_X = 0.0, Box_Y = 0.0;
  int fps = 60;
  int frame = 0, max_frame = 0;
  int window_size = 750;

  bool update_frame = true;
  std::chrono::system_clock::time_point current_time = std::chrono::system_clock::now();
  std::chrono::system_clock::time_point current_save_time = std::chrono::system_clock::now();

  bool auto_play = false;
  bool replay = false;
};

#endif