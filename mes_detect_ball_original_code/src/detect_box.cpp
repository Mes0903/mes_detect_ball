#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include "nav_msgs/Odometry.h"
#include "ros/ros.h"
#include "sensor_msgs/Imu.h"
#include "sensor_msgs/LaserScan.h"
#include "std_msgs/String.h"

#include "adaboost_classifier.h"
#include "load_matrix.h"
#include "make_feature.h"
#include "weak_learner.h"
#include "segment.h"
#include "normalize.h"
#include "file_handler.h"

#include <iostream>
#include <Eigen/Dense>
#include <limits>

Normalizer normalizer;
Adaboost A;

visualization_msgs::Marker marker;
uint32_t shape = visualization_msgs::Marker::CYLINDER;
ros::Publisher marker_pub;
ros::Publisher markerArray_pub;

void init_marker()
{
  // Initialize maker's setting.
  // Set the frame ID and timestamp.  See the TF tutorials for information on these.
  marker.header.frame_id = "base_scan";
  marker.header.stamp = ros::Time::now();

  // Set the namespace and id for this marker.  This serves to create a unique ID
  // Any marker sent with the same namespace and id will overwrite the old one
  marker.ns = "basic_shapes";
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
  marker.scale.x = 0.1;
  marker.scale.y = 0.1;
  marker.scale.z = 0.2;

  // Set the color -- be sure to set alpha to something non-zero!
  marker.color.r = 0.0f;
  marker.color.g = 1.0f;
  marker.color.b = 0.0f;
  marker.color.a = 1.0;

  // Tag(LIFETIME)
  marker.lifetime = ros::Duration();
}

void callback1(const ros::TimerEvent &)
{
  static int counter = 0;

  // update maker location and publish it.
  float x = 1 * std::cos(0.174 * counter);
  float y = 1 * std::sin(0.174 * counter);
  marker.pose.position.x = x;
  marker.pose.position.y = y;
  counter++;

  marker_pub.publish(marker);
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

  data = normalizer.transform(data);

  // 切分段
  const auto [feature_matrix, segment_vec] = transform_to_feature(data); // segment_vec is std::vector<Eigen::MatrixXd>

  // prediction
  puts("make prediction");
  Eigen::VectorXd pred_Y = A.predict(feature_matrix); // input 等等是 xy，所以要轉 feature
  for (int i = 0; i < pred_Y.size(); ++i)
  {
    if (pred_Y(i) == 1)
    {
      const Eigen::MatrixXd &M = segment_vec[i];
      std::cout << M(0, 0) << '\n';
    }
  }

  for (long unsigned int i = 0; i < markerArray.markers.size(); ++i)
    markerArray.markers[i].id = i;

  markerArray_pub.publish(markerArray);
}

int main([[maybe_unused]] int argc, char **argv)
{
  const std::string filepath = get_filepath(argv[0]);
  Weight_handle::load_weight(filepath + "/include/weight_data/adaboost_ball_weight.txt", A, normalizer);

  ros::init(argc, argv, "Detect_Box_Node");

  ros::NodeHandle n;
  ros::Subscriber sub = n.subscribe<sensor_msgs::LaserScan>("/scan", 1000, scanCallback);

  ros::Timer timer1 = n.createTimer(ros::Duration(0.1), callback1);

  marker_pub = n.advertise<visualization_msgs::Marker>("visualization_marker", 1);

  markerArray_pub = n.advertise<visualization_msgs::MarkerArray>("visualization_markerArray", 1000);

  init_marker();
  ros::spin();
}