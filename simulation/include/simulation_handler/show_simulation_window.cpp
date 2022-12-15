#include "imgui_header.h"
#include "show_simulation_window.h"

#include "make_feature.h"
#include "adaboost.h"
#include "logistic.h"
#include "segment.h"
#include "normalize.h"
#include "file_handler.h"

#include <Eigen/Eigen>
#include <chrono>
#include <thread>

extern Adaboost<logistic> A_box, A_ball;
extern Normalizer normalizer_ball, normalizer_box;
extern Eigen::MatrixXd xy_data;

static double Ball_X{}, Ball_Y{}, Box_X{}, Box_Y{};
static int simulation_window_size = 750;
extern int frame, max_frame;
static int fps = 60;

void ShowSimulationInformation()
{
  if (ImGui::TreeNodeEx("Simulation Information")) {
    float spacing = ImGui::GetStyle().ItemInnerSpacing.x;
    ImGui::Text("Ball is at: [%f, %f]", Ball_X, Ball_Y);
    ImGui::Text("Box is at: [%f, %f]", Box_X, Box_Y);

    /*----------Simulation Window Size----------*/
    ImGui::PushButtonRepeat(true);

    if (ImGui::ArrowButton("simulation_window_size_left", ImGuiDir_Left) && simulation_window_size > 100)
      --simulation_window_size;

    ImGui::SameLine(0.0f, spacing);
    if (ImGui::ArrowButton("simulation_window_size_right", ImGuiDir_Right) && simulation_window_size < 2000)
      ++simulation_window_size;

    ImGui::PopButtonRepeat();
    ImGui::SameLine(0.0f, spacing);
    ImGui::SliderInt("Simulation Window size", &simulation_window_size, 100, 2000, "%d");

    /*----------Auto play and Replay----------*/
    static bool auto_play = true;
    static bool replay = true;

    ImGui::Checkbox("Auto Play", &auto_play);
    if (auto_play && frame < max_frame)
      ++frame;

    ImGui::SameLine();
    ImGui::Checkbox("Replay", &replay);
    if (replay && frame >= max_frame)
      frame = 0;

    ImGui::Text("Max Frame: %d", max_frame);

    /*----------Frame Control----------*/
    ImGui::PushButtonRepeat(true);

    if (ImGui::ArrowButton("frame_left", ImGuiDir_Left) && !auto_play && frame > 0)
      --frame;

    ImGui::SameLine(0.0f, spacing);
    if (ImGui::ArrowButton("frame_right", ImGuiDir_Right) && !auto_play && frame < max_frame)
      ++frame;

    ImGui::PopButtonRepeat();
    ImGui::SameLine(0.0f, spacing);
    ImGui::SliderInt("Frame", &frame, 0, max_frame, "%d");
    ImGui::SameLine();
    ImGui::Text(": %d", frame);

    /*----------FPS Control----------*/
    ImGui::PushButtonRepeat(true);

    if (ImGui::ArrowButton("fps_left", ImGuiDir_Left) && !auto_play && fps > 1)
      --fps;

    ImGui::SameLine(0.0f, spacing);
    if (ImGui::ArrowButton("fps_right", ImGuiDir_Right) && !auto_play && fps < 200)
      ++fps;

    ImGui::SameLine(0.0f, spacing);
    ImGui::SliderInt("FPS", &fps, 1, 200, "%d");

    ImGui::PopButtonRepeat();
    ImGui::SameLine();
    ImGui::Text(": %d", fps);

    ImGui::TreePop();
  }
}

void ShowSimulation()
{
  ImGui::SetNextWindowPos(ImVec2(50, 50), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(500, 500), ImGuiCond_FirstUseEver);
  ImGui::Begin("Simulation Window");

  ShowSimulationInformation();

  if (ImGui::TreeNodeEx("Simulation window")) {
    if (ImPlot::BeginPlot("Simulation", ImVec2(simulation_window_size, simulation_window_size))) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1000 / fps));

      ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 1);
      ImPlot::SetupAxes("x", "y");
      ImPlot::SetupAxisLimits(ImAxis_X1, -5.0, 5.0);
      ImPlot::SetupAxisLimits(ImAxis_Y1, -10, 10);

      double order_using_xy = 2000;
      ImPlot::PlotScatter("Normal Point", &order_using_xy, &order_using_xy, 1);
      ImPlot::PlotScatter("Ball Section", &order_using_xy, &order_using_xy, 1);

      auto [feature_matrix, segment_vec] = section_to_feature(xy_data);    // segment_vec is std::vector<Eigen::MatrixXd>

      // Eigen::MatrixXd feature_matrix_box = normalizer_box.transform(feature_matrix);
      Eigen::MatrixXd feature_matrix_ball = normalizer_ball.transform(feature_matrix);

      // Eigen::VectorXd pred_Y_box = A_box.predict(feature_matrix_box);
      Eigen::VectorXd pred_Y_ball = A_ball.predict(feature_matrix_ball);

      for (std::size_t i = 0; i < segment_vec.size(); ++i) {
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 1, ImVec4(0, 0.7f, 0, 1), IMPLOT_AUTO, ImVec4(0, 0.7f, 0, 1));

        Eigen::ArrayXd section_x_data = segment_vec[i].col(0).array();
        Eigen::ArrayXd section_y_data = segment_vec[i].col(1).array();

        double *section_x = section_x_data.data();
        double *section_y = section_y_data.data();

        if (std::sqrt(std::pow(section_x_data.mean(), 2) + std::pow(section_y_data.mean(), 2)) < 1.5) {
          // if (pred_Y_box(i) == 1) {
          //   ImPlot::SetNextMarkerStyle(ImPlotMarker_Square, 3, ImVec4(1, 1, 0, 1), IMPLOT_AUTO, ImVec4(1, 1, 0, 1));
          //   ImGui::Text("Box is at: [%f, %f]", section_x_data.mean(), section_y_data.mean());
          //   ImPlot::PlotScatter("Box Section", section_x, section_y, section_x_data.size());
          // }


          if (pred_Y_ball(i) == 1) {
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 1, ImVec4(1, 0, 0, 1), IMPLOT_AUTO, ImVec4(1, 0, 0, 1));
            ImPlot::PlotScatter("Ball Section", section_x, section_y, section_x_data.size());

            Ball_X = section_x_data.mean();
            Ball_Y = section_y_data.mean();
          }
          else {
            ImPlot::PlotScatter("Normal Point", section_x, section_y, section_x_data.size());
          }
        }
        else {
          ImPlot::PlotScatter("Normal Point", section_x, section_y, section_x_data.size());
        }
      }

      ImPlot::PopStyleVar();
      ImPlot::EndPlot();
    }
    ImGui::TreePop();
  }

  ImGui::End();
}