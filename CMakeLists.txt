cmake_minimum_required(VERSION 3.0.2)
project(mes_detect_ball)

# C++ edition : C++17
set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# On windwos, it need the compile flag -I\"D:\\Eigen\", in linux it can find package
if(WIN32)
  set(EIGEN3_INCLUDE_DIRS D:/Eigen) # SET UR EIGEN Directory
  if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    MESSAGE("==================== USING MSVC TO COMILE ====================")
    set(CMAKE_CXX_FLAGS_DEBUG "/O2")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
  else()
    MESSAGE("==================== USING MINGW TO COMILE ====================")
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wa,-mbig-obj") # mingw compile flag (the output was weird idk why).
    set(CMAKE_CXX_FLAGS_DEBUG "-O3")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
  endif()
else()
  set(CMAKE_CXX_FLAGS "-Wall -Wextra")
  set(CMAKE_CXX_FLAGS_DEBUG "-g -O3")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif(WIN32)


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
aux_source_directory(./class_def CLASS_DEF)

# 
add_executable(Training_Ball
  ./src/training_ball.cpp
  ${CLASS_DEF}
)

target_link_libraries(
  Training_Ball
  ${catkin_LIBRARIES}
  ${Eigen3_LIBRARIES}
)

add_executable(Training_Box
  ./src/training_box.cpp
  ${CLASS_DEF}
)

target_link_libraries(
  Training_Box
  ${catkin_LIBRARIES}
  ${Eigen3_LIBRARIES}
)

# for ROS detection

if(UNIX)
  add_executable(Detect_Ball
    ./src/detect_ball.cpp
    ${CLASS_DEF}
  )

  target_link_libraries(
    Detect_Ball
    ${catkin_LIBRARIES}
    ${Eigen3_LIBRARIES}
  )

  add_executable(Detect_Box
  ./src/detect_box.cpp
  ${CLASS_DEF}
  )

  target_link_libraries(
    Detect_Box
    ${catkin_LIBRARIES}
    ${Eigen3_LIBRARIES}
  )
endif(UNIX)