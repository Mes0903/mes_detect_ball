cmake_minimum_required(VERSION 3.11)
 
project(DETECTION_SIMULATION)

set(CMAKE_CXX_STANDARD 17)

set(SIMULATION_DIR ${PROJECT_SOURCE_DIR})
set(SIMULATION_THIRD_DIR ${SIMULATION_DIR}/3rdparty)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PROJECT_HEADER}
  ${MODEL_DIR}
  ${MODEL_DIR}/adaboost
  ${MODEL_DIR}/logistic
  ${SIMULATION_THIRD_DIR}/imgui
  ${SIMULATION_THIRD_DIR}/implot
)

add_subdirectory(${SIMULATION_THIRD_DIR})

add_executable(
  DETECTION_SIMULATION 
  ${SIMULATION_DIR}/src/simulation.cpp
  ${MODEL_DIR}/normalize.h
  ${MODEL_DIR}/normalize.cpp
  ${MODEL_DIR}/adaboost/adaboost.h
  ${MODEL_DIR}/logistic/logistic.h
  ${MODEL_DIR}/logistic/logistic.cpp
)

if(WIN32)
  target_link_libraries(
    DETECTION_SIMULATION
    IMGUI_LIB
    IMPLOT_LIB
    ${SIMULATION_THIRD_DIR}/windows/lib/glfw3.lib
    ${SIMULATION_THIRD_DIR}/windows/lib/OpenGL32.Lib
  )

elseif(UNIX)
  find_package(OpenGL REQUIRED)
  if(OPENGL_FOUND)
    message('OPENGL_FOUND-is-true')
  else()
    message('OPENGL_FOUND-is-false')
  endif()

  target_link_libraries(
    DETECTION_SIMULATION
    IMGUI_LIB
    IMPLOT_LIB
    glfw
    ${OPENGL_LIBRARIES}
  )

endif()























# add_custom_target(Makefile COMMAND make
#                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
# )
