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

set(THIRD_PARTY_DIR ${PROJECT_SOURCE_DIR}/include/header)
set(DATA_SETS ${PROJECT_SOURCE_DIR}/include/dataset)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${THIRD_PARTY_DIR}
  ${DATA_SETS}
)

aux_source_directory(./lib_def LIB_DEF)
add_executable(Training_Ball
  ./src/training_ball.cpp
  ${LIB_DEF}
)

target_link_libraries(
  Training_Ball
  ${catkin_LIBRARIES}
  ${Eigen3_LIBRARIES}
)

add_executable(Detect_Ball
  ./src/main.cpp
  ${LIB_DEF}
)

target_link_libraries(
  Detect_Ball
  ${catkin_LIBRARIES}
  ${Eigen3_LIBRARIES}
)

add_executable(Training_BOX
  ./src/training_box.cpp
  ${LIB_DEF}
)

target_link_libraries(
  Detect_Ball
  ${catkin_LIBRARIES}
  ${Eigen3_LIBRARIES}
)

add_executable(Detect_BOX
  ./src/detect_box.cpp
  ${LIB_DEF}
)

target_link_libraries(
  Detect_BOX
  ${catkin_LIBRARIES}
  ${Eigen3_LIBRARIES}
)