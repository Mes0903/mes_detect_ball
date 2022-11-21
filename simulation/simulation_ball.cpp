/**
 * @file detect_ball.cpp
 * @author Mes (mes900903@gmail.com) (Discord: Mes#0903)
 * @brief Detect the ball by minibots, this file is a ROS node.
 *        Execute it by command `rosrun mes_detect_ball Detect_Ball` if you use ROS to build it.
 * @version 0.1
 * @date 2022-11-18
 */

#include "adaboost_classifier.h"
#include "make_feature.h"
#include "weak_learner.h"
#include "segment.h"
#include "normalize.h"
#include "file_handler.h"

#include <iostream>
#include <Eigen/Eigen>
#include <limits>

static Adaboost A;
static Normalizer normalizer;

void scanCallback(const Eigen::MatrixXd& scan_data)
{
  visualization_msgs::MarkerArray markerArray;
  
  const int ROW = 720;
  Eigen::MatrixXd data(ROW, 2);

  for (int i = 0; i < ROW; i++)
  {
    data(i, 0) = scan->ranges[i] * std::cos(scan->angle_min + scan->angle_increment * i);
    data(i, 1) = scan->ranges[i] * std::sin(scan->angle_min + scan->angle_increment * i);
  }

  // 切分段
  auto [feature_matrix, segment_vec] = section_to_feature(data); // segment_vec is std::vector<Eigen::MatrixXd>

  feature_matrix = normalizer.transform(feature_matrix);

  // prediction
  Eigen::VectorXd pred_Y = A.predict(feature_matrix); // input 等等是 xy，所以要轉 feature
  bool detect_flag = false;

  for (int i = 0; i < pred_Y.size(); ++i)
  {
    if (pred_Y(i) == 1)
    {
      detect_flag = true;

      const Eigen::MatrixXd &M = segment_vec[i].colwise().mean();

      if(std::sqrt(std::pow(M(0, 0), 2) + std::pow(M(0, 1), 2)) < 1.5) {

        std::cout << "Ball is at: [" << M(0, 0) << ", " << M(0, 1) << "]\n";

        marker.pose.position.x = M(0, 0);
        marker.pose.position.y = M(0, 1);
        marker.header.stamp = scan->header.stamp;
        markerArray.markers.push_back(marker);
      }
    }
  }

  for (long unsigned int i = 0; i < markerArray.markers.size(); ++i)
    markerArray.markers[i].id = i;

  markerArray_pub.publish(markerArray);

  if (detect_flag)
    std::cout << "---------------------------------\n";
}

int main([[maybe_unused]] int argc, char **argv)
{
  std::cout << "Detect Ball Node wake up\n";

  const std::string filepath = get_filepath(argv[0]);
  filepath += "/include/dataset/ball_xy_data.txt";

  constepxr int ROW = 720;
  Eigen::MatrixXd total_data = 
}