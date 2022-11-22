/**
 * @file detect_ball.cpp
 * @author Mes (mes900903@gmail.com) (Discord: Mes#0903)
 * @brief Detect the ball by minibots, this file is a ROS node.
 *        Execute it by command rosrun mes_detect_ball Detect_Ball if you use ROS to build it.
 * @version 0.1
 * @date 2022-11-18
 */

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include "nav_msgs/Odometry.h"
#include "ros/ros.h"
#include "sensor_msgs/Imu.h"
#include "sensor_msgs/LaserScan.h"
#include "std_msgs/String.h"

#include "adaboost_classifier.h"
#include "make_feature.h"
#include "weak_learner.h"
#include "segment.h"
#include "normalize.h"
#include "file_handler.h"
#include "bayes_filter.h"

#include <iostream>
#include <Eigen/Dense>
#include <limits>

Adaboost A_ball, A_box;
Normalizer normalizer_ball, normalizer_box;
Eigen::MatrixXd ball_buffer = Eigen::MatrixXd::Zero(1, 3); ////////
bool empty_buffer = true;
//Eigen::MatrixXd box_buffer = Eigen::MatrixXd::Zero(1, 3); ////////

visualization_msgs::Marker marker;
uint32_t shape = visualization_msgs::Marker::SPHERE;
ros::Publisher markerArray_pub;

void init_marker()
{
  // Initialize maker's setting.
  // Set the frame ID and timestamp.  See the TF tutorials for information on these.
  marker.header.frame_id = "laser_link";
  marker.header.stamp = ros::Time::now();

  // Set the namespace and id for this marker.  This serves to create a unique ID
  // Any marker sent with the same namespace and id will overwrite the old one
  marker.ns = "Detected_ball";
  marker.id = 0;
  // Set the marker type.  Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
  marker.type = shape;

  // Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
  // Tag(ACTION)
  marker.action = visualization_msgs::Marker::ADD;

  // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
  // Tag(POSE)
  marker.pose.position.x = 0;
  marker.pose.position.y = 0;
  marker.pose.position.z = 0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;

  // Set the scale of the marker -- 1x1x1 here means 1m on a side
  marker.scale.x = 0.2;
  marker.scale.y = 0.2;
  marker.scale.z = 0.2;

  // Set the color -- be sure to set alpha to something non-zero!
  marker.color.r = 1.0f;
  marker.color.g = 0.0f;
  marker.color.b = 0.0f;
  marker.color.a = 1.0;

  // Tag(LIFETIME)
  marker.lifetime = ros::Duration(0.15);
}

void scanCallback(const sensor_msgs::LaserScan::ConstPtr &scan)
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

  feature_matrix_ball = normalizer_ball.transform(feature_matrix);
  feature_matrix_box = normalizer_box.transform(feature_matrix);

  // prediction
  Eigen::VectorXd pred_Y_ball = A_ball.predict(feature_matrix_ball); // input 等等是 xy，所以要轉 feature
  Eigen::VectorXd pred_Y_box = A_box.predict(feature_matrix_box); // input 等等是 xy，所以要轉 feature
  bool detect_flag = false;
  
  // Bayes filter parameters
  Eigen::MatrixXd T(2,2);
  T << 0.8, 0.2, 0.2, 0.8;
  Eigen::MatrixXd Z_ball(2,2);
  Z_ball << 0.968, 0.032, 0.0002, 0.9998;
  
  int p1 = 0.014;
  int p0 = 0.986;
  ball_buffer = Bayes_filter(T, Z_ball, p1, p0, data, ball_buffer, empty_buffer);
  //Eigen::MatrixXd Z_box(2,2);
  //Z_box << 0.986, 0.014, 0.0001, 0.9999;
  //box_buffer = Bayes_filter(T, Z_box, p1, p0, data, box_buffer);
  
  for (int i = 0; i < pred_Y.size(); ++i)
  {
    if (pred_Y_box(i) == 1)
    {
      detect_flag = true;

      const Eigen::MatrixXd &M = segment_vec[i].colwise().mean();

      if(std::sqrt(std::pow(M(0, 0), 2) + std::pow(M(0, 1), 2)) < 1.5) {
        std::cout << "### " << ball_buffer << "\n";
        //std::cout << "Ball is at: [" << M(0, 0) << ", " << M(0, 1) << ", " << ball_buffer.col(2).maxCoeff() <<"]\n"; ///把球拿走會出事

        marker.pose.position.x = M(0, 0);
        marker.pose.position.y = M(0, 1);
        marker.ns = "Detected_box";
        marker.id = 1;
        marker.type = visualization_msgs::Marker::CUBE
        marker.header.stamp = scan->header.stamp;
        markerArray.markers.push_back(marker);
      }
    }
    
    if (pred_Y_ball(i) == 1)
    {
      detect_flag = true;

      const Eigen::MatrixXd &M = segment_vec[i].colwise().mean();

      if(std::sqrt(std::pow(M(0, 0), 2) + std::pow(M(0, 1), 2)) < 1.5) {
        std::cout << "### " << ball_buffer << "\n";
        //std::cout << "Ball is at: [" << M(0, 0) << ", " << M(0, 1) << ", " << ball_buffer.col(2).maxCoeff() <<"]\n"; ///把球拿走會出事

        marker.pose.position.x = M(0, 0);
        marker.pose.position.y = M(0, 1);
        marker.ns = "Detected_ball";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::SPHERE
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

  Weight_handle::load_weight(filepath + "/include/weight_data/adaboost_ball_weight.txt", A_ball, normalizer_ball);
  Weight_handle::load_weight(filepath + "/include/weight_data/adaboost_box_weight.txt", A_box, normalizer_box);

  ros::init(argc, argv, "Detect_Ball_Node");

  ros::NodeHandle n;

  ros::Subscriber sub = n.subscribe<sensor_msgs::LaserScan>("/scan", 1, scanCallback);

  markerArray_pub = n.advertise<visualization_msgs::MarkerArray>("Ball_MarkerArray", 1);

  init_marker();

  ros::spin();
}