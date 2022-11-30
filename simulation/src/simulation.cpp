// Sample app built with Dear ImGui and ImPlot
// This app uses implot and imgui, but does not output to any backend! It only serves as a proof that an app can be built, linked, and run.

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include <chrono>
#include <thread>
#include <math.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h>    // Will drag system OpenGL headers

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif


double box_x_data[720], box_y_data[720];

void Plot_point()
{
  if (ImPlot::BeginPlot("Scatter Plot", ImVec2(750, 750))) {
    ImPlot::SetupAxes("x", "y");
    ImPlot::SetupAxisLimits(ImAxis_X1, -5.0, 5.0);
    ImPlot::SetupAxisLimits(ImAxis_Y1, -10, 10);

    ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.1f);
    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 1, ImVec4(0, 1, 0, 0.5f), IMPLOT_AUTO, ImVec4(0, 1, 0, 1));
    ImPlot::PlotScatter("Data 1", box_x_data, box_y_data, 720);
    ImPlot::PopStyleVar();
    ImPlot::EndPlot();
  }
}

static void glfw_error_callback(int error, const char *description)
{
  fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

int main(int, char **)
{
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit())
    return 1;

    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
  // GL ES 2.0 + GLSL 100
  const char *glsl_version = "#version 100";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
  // GL 3.2 + GLSL 150
  const char *glsl_version = "#version 150";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);    // 3.2+ only
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);    // Required on Mac
#else
  // GL 3.0 + GLSL 130
  const char *glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
  //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

  // Create window with graphics context
  GLFWwindow *window = glfwCreateWindow(1000, 800, "Detection simulation", NULL, NULL);
  if (window == NULL)
    return 1;
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);    // Enable vsync

  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void) io;
  //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
  //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();
  //ImGui::StyleColorsLight();

  // Setup Platform/Renderer backends
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  bool show_simulation_window = true;
  bool show_debug_window = true;
  ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

  /*-----------------------------------------------------------------------------------------------------*/
#if _WIN32
  const std::string filepath = "D:/document/GitHub/mes_detect_ball/include/dataset/box_xy_data.txt";
#else
  const std::string filepath = "/home/mes/mes_detect_ball/include/dataset/box_xy_data.txt";
#endif
  std::ifstream infile(filepath);
  if (infile.fail()) {
    std::cout << "cant found " << filepath << '\n';
    exit(1);
  }

  std::string line;
  std::stringstream ss;
  int cnt = 0;
  while (!glfwWindowShouldClose(window)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    glfwPollEvents();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    for (int i{}; i < 720; ++i) {
      std::getline(infile, line);
      ss << line;
      ss >> box_x_data[i] >> box_y_data[i];
      ss.str("");
      ss.clear();
    }

    if (infile.eof()) {
      infile.clear();
      infile.seekg(0, std::ios::beg);
    }

    if (show_simulation_window) {
      ImGui::SetNextWindowPos(ImVec2(50, 50), ImGuiCond_FirstUseEver);
      ImGui::SetNextWindowSize(ImVec2(750, 750), ImGuiCond_FirstUseEver);
      ImGui::Begin("Simulation Window", &show_simulation_window);    // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)

      Plot_point();

      ImGui::End();
    }

    if (!show_debug_window) {
      ImGui::SetNextWindowPos(ImVec2(100, 50), ImGuiCond_FirstUseEver);
      ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
      ImGui::Begin("Debug Window", &show_debug_window);
      ImGui::Text("counter = %d", cnt);
      ImGui::Text("line = %s", line);
      ImGui::Text("box_xy = [%f, %f]", box_x_data[0], box_y_data[0]);
      ImGui::End();
    }

    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }

  // Cleanup
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImPlot::DestroyContext();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}