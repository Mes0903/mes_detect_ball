# C++ edition : C++17
set(CMAKE_CXX_STANDARD 17)

# for ROS detection
if(UNIX)
  add_executable(Detect_Ball
    detect_ball.cpp
    ${LIB_DEF}
  )

  target_link_libraries(
    Detect_Ball
    ${catkin_LIBRARIES}
    ${Eigen3_LIBRARIES}
  )

  add_executable(Detect_BOX
    detect_box.cpp
    ${LIB_DEF}
  )

  target_link_libraries(
    Detect_BOX
    ${catkin_LIBRARIES}
    ${Eigen3_LIBRARIES}
  )
endif(UNIX)