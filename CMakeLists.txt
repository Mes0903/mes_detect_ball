cmake_minimum_required(VERSION 3.0.2)
project(mes_detect_ball)

# C++ edition : C++20
set(CMAKE_CXX_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# On windwos, it need the compile flag -I\"D:\\Eigen\", in linux it can find package
if(WIN32)
  set(CMAKE_CXX_FLAGS "-I\"D:\\Eigen\" -Wall -Wextra")
else()
  set(CMAKE_CXX_FLAGS "-Wall -Wextra")
endif(WIN32)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# For ROS and catkin_make
if(UNIX)
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
endif(UNIX)

# Project header
set(THIRD_PARTY_DIR ${PROJECT_SOURCE_DIR}/include/header)
set(DATA_SETS ${PROJECT_SOURCE_DIR}/include/dataset)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${THIRD_PARTY_DIR}
  ${DATA_SETS}
)

# Definition of headers
aux_source_directory(./lib_def LIB_DEF)

# 
add_executable(Training_Ball
  ./src/training_ball.cpp
  ${LIB_DEF}
)

target_link_libraries(
  Training_Ball
  ${catkin_LIBRARIES}
  ${Eigen3_LIBRARIES}
)

add_executable(Training_BOX
  ./src/training_box.cpp
  ${LIB_DEF}
)

target_link_libraries(
  Training_BOX
  ${catkin_LIBRARIES}
  ${Eigen3_LIBRARIES}
)

# for ROS detection

if(__WIN32__)
  add_executable(Detect_Ball
    ./src/main.cpp
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
endif(__WIN32__)