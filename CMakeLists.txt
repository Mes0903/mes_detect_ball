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
    find_package(OpenGL REQUIRED)
    MESSAGE("==================== USING MSVC TO COMILE ====================")
    add_compile_options(/wd4819 /wd4244 /wd4267 /wd4305)
    set(APP_ICON_RESOURCE_WINDOWS "${CMAKE_CURRENT_SOURCE_DIR}/icon/MesIcon.rc")
    set(CMAKE_CXX_FLAGS_DEBUG "/O2")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
  else()
    MESSAGE("==================== USING MINGW TO COMILE ====================")
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wa,-mbig-obj") # mingw compile flag (the output was weird idk why).
    set(CMAKE_CXX_FLAGS_DEBUG "-O3")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
  endif()
else()
  find_package(Eigen3 REQUIRED)
  set(CMAKE_CXX_FLAGS "-Wall -Wextra")
  set(CMAKE_CXX_FLAGS_DEBUG "-g -O3")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif(WIN32)


# For ROS and catkin_make
#if(UNIX)
#  find_package(Eigen3 REQUIRED)
#  find_package(catkin REQUIRED COMPONENTS
#    roscpp
#    rospy
#    std_msgs
#  )
#
#  catkin_package(
#    INCLUDE_DIRS include
#    LIBRARIES mes_detect_ball
#    CATKIN_DEPENDS roscpp rospy std_msgs
#    DEPENDS EIGEN3
#  )
#endif(UNIX)

# Project header
set(PROJECT_HEADER ${PROJECT_SOURCE_DIR}/include)
set(MODEL_DIR ${PROJECT_SOURCE_DIR}/Model)
set(THIRD_DIR ${PROJECT_SOURCE_DIR}/3rdparty)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PROJECT_HEADER}
  ${MODEL_DIR}
  ${MODEL_DIR}/adaboost
  ${MODEL_DIR}/logistic
  ${THIRD_DIR}/imgui
  ${THIRD_DIR}/implot
  ${SIMULATION_DIR}/include/
  ${SIMULATION_DIR}/include/windows_handler
  ${SIMULATION_DIR}/include/label_handler
  ${SIMULATION_DIR}/include/simulation_handler
)

 
add_executable(Training_Ball
  ./src/training_ball.cpp

  ${PROJECT_HEADER}/file_handler.h
  ${PROJECT_HEADER}/file_handler.cpp
  ${PROJECT_HEADER}/make_feature.h
  ${PROJECT_HEADER}/make_feature.cpp
  ${PROJECT_HEADER}/metric.h
  ${PROJECT_HEADER}/metric.cpp

  ${MODEL_DIR}/normalize.h
  ${MODEL_DIR}/normalize.cpp
  ${MODEL_DIR}/adaboost/adaboost.h
  ${MODEL_DIR}/logistic/logistic.h
  ${MODEL_DIR}/logistic/logistic.cpp

  $<$<BOOL:${WIN32}>: ${APP_ICON_RESOURCE_WINDOWS}>
)

target_link_libraries(
  Training_Ball
  ${catkin_LIBRARIES}
  ${Eigen3_LIBRARIES}
)

add_executable(Training_Box
  ./src/training_box.cpp

  ${PROJECT_HEADER}/file_handler.h
  ${PROJECT_HEADER}/file_handler.cpp
  ${PROJECT_HEADER}/make_feature.h
  ${PROJECT_HEADER}/make_feature.cpp
  ${PROJECT_HEADER}/metric.h
  ${PROJECT_HEADER}/metric.cpp

  ${MODEL_DIR}/normalize.h
  ${MODEL_DIR}/normalize.cpp
  ${MODEL_DIR}/adaboost/adaboost.h
  ${MODEL_DIR}/logistic/logistic.h
  ${MODEL_DIR}/logistic/logistic.cpp

  $<$<BOOL:${WIN32}>: ${APP_ICON_RESOURCE_WINDOWS}>
)

target_link_libraries(
  Training_Box
  ${catkin_LIBRARIES}
  ${Eigen3_LIBRARIES}
)

add_subdirectory(${THIRD_DIR})
add_subdirectory(${PROJECT_SOURCE_DIR}/simulation)

# for ROS detection

#if(UNIX)
#  add_executable(Detect_node
#    ./src/detect_node.cpp
#    ${CLASS_DEF}
#  )
#
#  target_link_libraries(
#    Detect_node
#    ${catkin_LIBRARIES}
#    ${Eigen3_LIBRARIES}
#  )
#endif(UNIX)