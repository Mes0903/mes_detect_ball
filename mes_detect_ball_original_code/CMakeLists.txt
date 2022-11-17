cmake_minimum_required(VERSION 3.0.2)
project(mes_detect_ball)

# C++ edition : C++17 (for ros, ros haven't supports C++20 now)
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
set(DETECTION_DIR ${PROJECT_SOURCE_DIR}/detection)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${THIRD_PARTY_DIR}
  ${DATA_SETS}
)

# Definition of headers
aux_source_directory(./lib_def LIB_DEF)

# Training executable file directories.
add_subdirectory(${DETECTION_DIR})

# Training executable file.
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