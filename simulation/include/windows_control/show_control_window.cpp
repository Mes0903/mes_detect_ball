#include "imgui_header.h"
#include "show_control_window.h"

void ShowControlWindow(bool &show_simulation_window)
{
  ImGui::Begin("Control window");
  ImGui::Checkbox("Show Simulation window", &show_simulation_window);
  ImGui::End();
}