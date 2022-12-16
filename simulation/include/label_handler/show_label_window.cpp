#include "imgui_header.h"
#include "show_label_window.h"
#include "frame_handler.h"

#include "make_feature.h"
#include "adaboost.h"
#include "logistic.h"
#include "segment.h"
#include "normalize.h"
#include "file_handler.h"

#include <Eigen/Eigen>
#include <chrono>
#include <thread>
#include <vector>
#include <unordered_set>
#include <algorithm>

extern Eigen::MatrixXd xy_data;

static double Ball_X{}, Ball_Y{}, Box_X{}, Box_Y{};
static int label_window_size = 750;
extern int frame, max_frame;
static int fps = 60;
static float label_mouse_area = static_cast<float>(0.05);

static bool update_frame = false;
static auto current_time = std::chrono::system_clock::now();
static bool auto_play = true;

void ShowLabelInformation()
{
  if (ImGui::TreeNodeEx("Label Information")) {
    float spacing = ImGui::GetStyle().ItemInnerSpacing.x;
    ImGui::Text("Ball is at: [%f, %f]", Ball_X, Ball_Y);
    ImGui::Text("Box is at: [%f, %f]", Box_X, Box_Y);

    /*----------Label Window Size----------*/
    ImGui::PushButtonRepeat(true);

    if (ImGui::ArrowButton("label_window_size_left", ImGuiDir_Left) && label_window_size > 100)
      --label_window_size;

    ImGui::SameLine(0.0f, spacing);
    if (ImGui::ArrowButton("label_window_size_right", ImGuiDir_Right) && label_window_size < 2000)
      ++label_window_size;

    ImGui::PopButtonRepeat();
    ImGui::SameLine(0.0f, spacing);
    ImGui::SliderInt("Label Window size", &label_window_size, 100, 2000, "%d");
    ImGui::SameLine();
    ImGui::Text(":%d", label_window_size);

    /*----------Auto play and Replay----------*/
    ImGui::Checkbox("Auto Play", &auto_play);
    if (auto_play && update_frame && frame < max_frame) {
      update_frame = true;
      ++frame;
    }

    ImGui::SameLine();

    static bool replay = true;
    ImGui::Checkbox("Replay", &replay);
    if (replay && update_frame && frame >= max_frame) {
      update_frame = true;
      frame = 0;
    }

    /*----------Label Mouse Area Control----------*/
    ImGui::PushButtonRepeat(true);

    if (ImGui::ArrowButton("label_mouse_area_left", ImGuiDir_Left) && !auto_play && label_mouse_area > 0) {
      update_frame = true;
      label_mouse_area -= static_cast<float>(0.01);
    }

    ImGui::SameLine(0.0f, spacing);
    if (ImGui::ArrowButton("label_mouse_area_right", ImGuiDir_Right) && !auto_play && label_mouse_area < 5) {
      update_frame = true;
      label_mouse_area += static_cast<float>(0.01);
    }

    ImGui::PopButtonRepeat();
    ImGui::SameLine(0.0f, spacing);
    ImGui::SliderFloat("Label Mouse Area", &label_mouse_area, 0, 5, "%0.01f");

    ImGui::SameLine();
    ImGui::Text(":%f", label_mouse_area);

    /*----------Frame Control----------*/
    ImGui::Text("Max Frame: %d", max_frame);
    ImGui::PushButtonRepeat(true);

    if (ImGui::ArrowButton("frame_left", ImGuiDir_Left) && !auto_play && frame > 0) {
      update_frame = true;
      --frame;
    }

    ImGui::SameLine(0.0f, spacing);
    if (ImGui::ArrowButton("frame_right", ImGuiDir_Right) && !auto_play && frame < max_frame) {
      update_frame = true;
      ++frame;
    }

    ImGui::PopButtonRepeat();
    ImGui::SameLine(0.0f, spacing);
    if (ImGui::SliderInt("Frame", &frame, 0, max_frame, "%d"))
      update_frame = true;

    ImGui::SameLine();
    ImGui::Text(":%d", frame);

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
    ImGui::Text(":%d", fps);

    ImGui::TreePop();
  }
}

// struct point_hash {
//   std::size_t operator()(const ImPlotPoint &point) const { return ((point.x + point.y) * (point.x + point.y + 1) / 2) + point.y; };
// };
// bool operator==(const ImPlotPoint &p1, const ImPlotPoint &p2) { return (p1.x == p2.x) && (p1.y == p2.y); };

void ShowLabel(std::ifstream &infile)
{
  ImGui::SetNextWindowPos(ImVec2(50, 50), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(500, 500), ImGuiCond_FirstUseEver);
  ImGui::Begin("Label Window");

  if (auto_play) {
    if (auto now = std::chrono::system_clock::now();
        std::chrono::duration_cast<std::chrono::milliseconds>(now - current_time).count() > (1000 / fps)) {
      current_time = now;
      update_frame = true;
    }
    else {
      update_frame = false;
    }
  }
  else {
    update_frame = false;
  }

  ShowLabelInformation();

  if (update_frame)
    read_frame(infile);

  if (ImGui::TreeNodeEx("Label window")) {
    if (ImPlot::BeginPlot("Label", ImVec2(static_cast<float>(label_window_size), static_cast<float>(label_window_size)))) {
      ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 1);
      ImPlot::SetupAxes("x", "y");
      ImPlot::SetupAxisLimits(ImAxis_X1, -5.0, 5.0);
      ImPlot::SetupAxisLimits(ImAxis_Y1, -10, 10);

      double order_using_xy = 0;
      ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 0, ImVec4(0, 0.7f, 0, 1), IMPLOT_AUTO, ImVec4(0, 0.7f, 0, 1));
      ImPlot::PlotScatter("Normal Point", &order_using_xy, &order_using_xy, 1);
      ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 0, ImVec4(1, 0, 0, 1), IMPLOT_AUTO, ImVec4(1, 0, 0, 1));
      ImPlot::PlotScatter("Ball Section", &order_using_xy, &order_using_xy, 1);

      auto [feature_matrix, segment_vec] = section_to_feature(xy_data);    // segment_vec is std::vector<Eigen::MatrixXd>

      // hash function get from https://stackoverflow.com/questions/682438/hash-function-providing-unique-uint-from-an-integer-coordinate-pair
      auto point_hash = [](const ImPlotPoint &point) { return ((point.x + point.y) * (point.x + point.y + 1) / 2) + point.y; };
      auto point_equal = [](const ImPlotPoint &p1, const ImPlotPoint &p2) { return (p1.x == p2.x) && (p1.y == p2.y); };
      static std::unordered_set<ImPlotPoint, decltype(point_hash), decltype(point_equal)> label_data{ 0, point_hash, point_equal };

      for (const Eigen::MatrixXd &segment : segment_vec) {
        Eigen::ArrayXd segment_x = segment.col(0).array();
        Eigen::ArrayXd segment_y = segment.col(1).array();
        double X_mean = segment_x.mean();
        double Y_mean = segment_y.mean();
        int segment_size = segment_x.size();

        if (ImPlot::IsPlotHovered() && ImGui::IsMouseClicked(0)) {
          ImPlotPoint click_point = ImPlot::GetPlotMousePos();

          for (auto data_point : segment.rowwise()) {
            if ((data_point(0) - label_mouse_area < click_point.x) && (click_point.x < data_point(0) + label_mouse_area) && (data_point(1) - label_mouse_area < click_point.y) && (click_point.y < data_point(1) + label_mouse_area)) {
              if (label_data.find(ImPlotPoint(X_mean, Y_mean)) != label_data.end())
                label_data.erase(ImPlotPoint(X_mean, Y_mean));
              else
                label_data.insert(ImPlotPoint(X_mean, Y_mean));

              break;
            }
          }
        }

        if (label_data.find(ImPlotPoint(X_mean, Y_mean)) != label_data.end()) {
          ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 1, ImVec4(1, 0, 0, 1), IMPLOT_AUTO, ImVec4(1, 0, 0, 1));
          ImPlot::PlotScatter("Ball Section", segment_x.data(), segment_y.data(), segment_size);
        }
        else {
          ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 1, ImVec4(0, 0.7f, 0, 1), IMPLOT_AUTO, ImVec4(0, 0.7f, 0, 1));
          ImPlot::PlotScatter("Normal Point", segment_x.data(), segment_y.data(), segment_size);
        }
      }

      ImPlot::PopStyleVar();
      ImPlot::EndPlot();
    }

    ImGui::TreePop();
  }

  ImGui::End();
}