cmake_minimum_required(VERSION 3.11)
project(MRL_LabelTool)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
  
set(PROJECT_HEADER ${CMAKE_SOURCE_DIR}/include)
set(RVIZ_DIR ${CMAKE_SOURCE_DIR}/rviz)
set(MODEL_DIR ${CMAKE_SOURCE_DIR}/Model)
set(EIGEN3_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/3rdparty/Eigen)

set(TRAINING_DIR ${CMAKE_SOURCE_DIR}/Training)

add_subdirectory(${TRAINING_DIR})
add_subdirectory(${RVIZ_DIR})