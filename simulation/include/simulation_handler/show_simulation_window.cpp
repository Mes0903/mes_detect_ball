#include "imgui_header.h"
#include "show_simulation_window.h"
#include "animate_info.h"

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

static AnimationInfo SI;    // simulation animation info

void ShowSimulationInformation()
{
  if (ImGui::TreeNodeEx("Simulation Information")) {
    float spacing = ImGui::GetStyle().ItemInnerSpacing.x;
    ImGui::Text("Ball is at: [%f, %f]", SI.Ball_X, SI.Ball_Y);
    ImGui::Text("Box is at: [%f, %f]", SI.Box_X, SI.Box_Y);

    /*----------Simulation Window Size----------*/
    ImGui::Text("Label Window Size Control:");
    ImGui::SameLine();
    ImGui::PushButtonRepeat(true);

    if (ImGui::ArrowButton("simulation_window_size_left", ImGuiDir_Left) && SI.window_size > 100)
      --SI.window_size;

    ImGui::SameLine(0.0f, spacing);
    if (ImGui::ArrowButton("simulation_window_size_right", ImGuiDir_Right) && SI.window_size < 2000)
      ++SI.window_size;

    ImGui::PopButtonRepeat();
    ImGui::SameLine(0.0f, spacing);
    ImGui::SliderInt("Simulation Window size", &SI.window_size, 100, 2000, "%d");
    ImGui::SameLine();
    ImGui::Text(":%d", SI.window_size);

    /*----------Frame Control----------*/
    ImGui::Text("Max Frame: %d", SI.max_frame);
    ImGui::PushButtonRepeat(true);

    ImGui::Text("Frame Control:");
    ImGui::SameLine();

    if (ImGui::ArrowButton("frame_left", ImGuiDir_Left) && !SI.auto_play && SI.frame > 0) {
      SI.update_frame = true;
      --SI.frame;
    }

    ImGui::SameLine(0.0f, spacing);
    if (ImGui::ArrowButton("frame_right", ImGuiDir_Right) && !SI.auto_play && SI.frame < SI.max_frame) {
      SI.update_frame = true;
      ++SI.frame;
    }

    ImGui::PopButtonRepeat();
    ImGui::SameLine(0.0f, spacing);
    if (ImGui::SliderInt("Frame", &SI.frame, 0, SI.max_frame, "%d"))
      SI.update_frame = true;

    ImGui::SameLine();
    ImGui::Text(": %d", SI.frame);

    /*----------FPS Control----------*/
    ImGui::Text("FPS Control:");
    ImGui::SameLine();
    ImGui::PushButtonRepeat(true);

    if (ImGui::ArrowButton("fps_left", ImGuiDir_Left) && SI.fps > 1)
      --SI.fps;

    ImGui::SameLine(0.0f, spacing);
    if (ImGui::ArrowButton("fps_right", ImGuiDir_Right) && SI.fps < 200)
      ++SI.fps;

    ImGui::SameLine(0.0f, spacing);
    ImGui::SliderInt("FPS", &SI.fps, 1, 200, "%d");

    ImGui::PopButtonRepeat();
    ImGui::SameLine();
    ImGui::Text(": %d", SI.fps);

    /*----------Auto play and Replay----------*/
    ImGui::Checkbox("Auto Play", &SI.auto_play);
    if (SI.auto_play && SI.update_frame && SI.frame < SI.max_frame) {
      SI.update_frame = true;
      ++SI.frame;
    }

    ImGui::SameLine();

    ImGui::Checkbox("Replay", &SI.replay);
    if (SI.replay && SI.update_frame && SI.frame >= SI.max_frame) {
      SI.update_frame = true;
      SI.frame = 0;
    }

    ImGui::TreePop();
  }
}

void ShowSimulation(std::ifstream &infile, const int max_frame)
{
  SI.max_frame = max_frame;

  ImGui::SetNextWindowPos(ImVec2(550, 50), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(500, 500), ImGuiCond_FirstUseEver);
  ImGui::Begin("Simulation Window");

  if (SI.auto_play) {
    if (auto now = std::chrono::system_clock::now();
        std::chrono::duration_cast<std::chrono::milliseconds>(now - SI.current_time).count() > (1000 / SI.fps)) {
      SI.current_time = now;
      SI.update_frame = true;
    }
    else {
      SI.update_frame = false;
    }
  }

  ShowSimulationInformation();

  static Eigen::MatrixXd feature_matrix;
  static std::vector<Eigen::MatrixXd> segment_vec;

  if (SI.update_frame) {
    SI.update_frame = false;

    File_handler::read_frame(infile, SI.xy_data, SI.frame);
    std::tie(feature_matrix, segment_vec) = section_to_feature(SI.xy_data);
  }

  if (ImGui::TreeNodeEx("Simulation window")) {
    if (ImPlot::BeginPlot("Simulation", ImVec2(static_cast<float>(SI.window_size), static_cast<float>(SI.window_size)))) {
      ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 1);
      ImPlot::SetupAxes("x", "y");
      ImPlot::SetupAxisLimits(ImAxis_X1, -5.0, 5.0);
      ImPlot::SetupAxisLimits(ImAxis_Y1, -10, 10);

      double order_using_xy = 0;
      ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 0, ImVec4(0, 0.7f, 0, 1), IMPLOT_AUTO, ImVec4(0, 0.7f, 0, 1));
      ImPlot::PlotScatter("Normal Point", &order_using_xy, &order_using_xy, 1);
      ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 0, ImVec4(1, 0, 0, 1), IMPLOT_AUTO, ImVec4(1, 0, 0, 1));
      ImPlot::PlotScatter("Ball Section", &order_using_xy, &order_using_xy, 1);

      // Eigen::MatrixXd feature_matrix_box = normalizer_box.transform(feature_matrix);
      Eigen::MatrixXd feature_matrix_ball = normalizer_ball.transform(feature_matrix);

      // Eigen::VectorXd pred_Y_box = A_box.predict(feature_matrix_box);
      Eigen::VectorXd pred_Y_ball = A_ball.predict(feature_matrix_ball);

      for (int i = 0; i < segment_vec.size(); ++i) {
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

          // ImGui::IsMouseClicked(i);
          if (pred_Y_ball(i) == 1) {
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 1, ImVec4(1, 0, 0, 1), IMPLOT_AUTO, ImVec4(1, 0, 0, 1));
            ImPlot::PlotScatter("Ball Section", section_x, section_y, section_x_data.size());

            SI.Ball_X = section_x_data.mean();
            SI.Ball_Y = section_y_data.mean();
          }
          else {
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 1, ImVec4(0, 0.7f, 0, 1), IMPLOT_AUTO, ImVec4(0, 0.7f, 0, 1));
            ImPlot::PlotScatter("Normal Point", section_x, section_y, section_x_data.size());
          }
        }
        else {
          ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 1, ImVec4(0, 0.7f, 0, 1), IMPLOT_AUTO, ImVec4(0, 0.7f, 0, 1));
          ImPlot::PlotScatter("Normal Point", section_x, section_y, section_x_data.size());
        }
      }

      ImPlot::PopStyleVar();
      ImPlot::EndPlot();
    }    // end ImPlot::BeginPlot("Simulation", ...)

    ImGui::TreePop();
  }

  ImGui::End();
}