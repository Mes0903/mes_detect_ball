cmake_minimum_required(VERSION 3.0.2)
project(Detect_node)

# C++ edition : C++17
set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 REQUIRED)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(Eigen3 REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES mes_detect_ball
  CATKIN_DEPENDS roscpp rospy std_msgs
  DEPENDS EIGEN3
)

add_executable(Detect_node
  ./src/detect_node.cpp
  ${CLASS_DEF}
)

target_link_libraries(
  Detect_node
  ${catkin_LIBRARIES}
  ${Eigen3_LIBRARIES}
)