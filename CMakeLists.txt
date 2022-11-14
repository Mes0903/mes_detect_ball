cmake_minimum_required(VERSION 3.0.2)
project(mes_detect_ball)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
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

set(THIRD_PARTY_DIR ${PROJECT_SOURCE_DIR}/include/mes_detect_ball)
set(DATA_SETS ${PROJECT_SOURCE_DIR}/include/dataset)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${THIRD_PARTY_DIR}
  ${DATA_SETS}
)

aux_source_directory(./src DIR_SRCS)
add_executable(Detect_node ${DIR_SRCS})

target_link_libraries(
  Detect_node
  ${catkin_LIBRARIES}
  ${Eigen3_LIBRARIES}
)
