#include "imgui_header.h"
#include "show_label_window.h"
#include "label_animate_info.h"

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
#include <set>
#include <algorithm>
#include <fstream>
#include <cstdio>
#include <filesystem>

static LabelAnimationInfo LI;

static ImVec4 color_arr[] = { ImVec4(192 / 255.0, 238 / 255.0, 228 / 255.0, 1),
                              ImVec4(248 / 255.0, 249 / 255.0, 136 / 255.0, 1),
                              ImVec4(255 / 255.0, 202 / 255.0, 200 / 255.0, 1),
                              ImVec4(255 / 255.0, 158 / 255.0, 158 / 255.0, 1),
                              ImVec4(250 / 255.0, 248 / 255.0, 241 / 255.0, 1),
                              ImVec4(250 / 255.0, 234 / 255.0, 177 / 255.0, 1),
                              ImVec4(229 / 255.0, 186 / 255.0, 115 / 255.0, 1),
                              ImVec4(197 / 255.0, 137 / 255.0, 64 / 255.0, 1),
                              ImVec4(204 / 255.0, 214 / 255.0, 166 / 255.0, 1) };

void ShowLabelInformation(std::fstream &feature_file, std::fstream &label_file)
{
  static const std::string filepath = File_handler::get_filepath();

  /*----------------------------------------*/


  if (ImGui::TreeNodeEx("Label Information")) {
    if (ImGui::Button("Clearing Data"))
      ImGui::OpenPopup("Clean?");

    if (ImGui::BeginPopupModal("Clean?", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
      ImGui::Text("Are you sure to clean up the data?\nAll of the data you wrote will be deleted.\n\n");
      ImGui::Separator();

      if (ImGui::Button("OK", ImVec2(200, 0))) {
        LI.clean_data = true;
        ImGui::CloseCurrentPopup();
      }

      ImGui::SetItemDefaultFocus();
      ImGui::SameLine();
      if (ImGui::Button("Cancel", ImVec2(200, 0))) { ImGui::CloseCurrentPopup(); }
      ImGui::EndPopup();
    }

    float spacing = ImGui::GetStyle().ItemInnerSpacing.x;
    ImGui::Text("Ball is at: [%f, %f]", LI.Ball_X, LI.Ball_Y);
    ImGui::Text("Box is at: [%f, %f]", LI.Box_X, LI.Box_Y);

    /*----------Label Window Size----------*/
    ImGui::Text("Label Window Size Control:");
    ImGui::SameLine();
    ImGui::PushButtonRepeat(true);

    if (ImGui::ArrowButton("label_window_size_left", ImGuiDir_Left) && LI.window_size > 100)
      --LI.window_size;

    ImGui::SameLine(0.0f, spacing);
    if (ImGui::ArrowButton("label_window_size_right", ImGuiDir_Right) && LI.window_size < 2000)
      ++LI.window_size;

    ImGui::PopButtonRepeat();
    ImGui::SameLine(0.0f, spacing);
    ImGui::SliderInt("Label Window size", &LI.window_size, 100, 2000, "%d");
    ImGui::SameLine();
    ImGui::Text(":%d", LI.window_size);

    /*----------Label Mouse Area Control----------*/
    ImGui::Text("Mouse Area Control:");
    ImGui::SameLine();
    ImGui::PushButtonRepeat(true);

    if (ImGui::ArrowButton("label_mouse_area_left", ImGuiDir_Left) && !LI.auto_play && LI.label_mouse_area > 0) {
      LI.update_frame = true;
      LI.label_mouse_area -= static_cast<float>(0.01);
    }

    ImGui::SameLine(0.0f, spacing);
    if (ImGui::ArrowButton("label_mouse_area_right", ImGuiDir_Right) && !LI.auto_play && LI.label_mouse_area < 5) {
      LI.update_frame = true;
      LI.label_mouse_area += static_cast<float>(0.01);
    }

    ImGui::PopButtonRepeat();
    ImGui::SameLine(0.0f, spacing);
    ImGui::SliderFloat("Label Mouse Area", &LI.label_mouse_area, 0, 5, "%0.01f");

    ImGui::SameLine();
    ImGui::Text(":%f", LI.label_mouse_area);

    /*----------Frame Control----------*/
    ImGui::Text("Max Frame: %d", LI.max_frame);
    ImGui::Text("Writed Max Frame: %d", LI.writed_max_frame);
    ImGui::Text("Writed Frame Numbers: %d", LI.writed_frame_numbers);

    ImGui::Text("Frame Control:");
    ImGui::SameLine();

    ImGui::PushButtonRepeat(true);

    if ((ImGui::ArrowButton("frame_left", ImGuiDir_Left) || ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) && !LI.auto_play && LI.frame > 0) {
      LI.update_frame = true;
      --LI.frame;
    }

    ImGui::SameLine(0.0f, spacing);
    if ((ImGui::ArrowButton("frame_right", ImGuiDir_Right) || ImGui::IsKeyPressed(ImGuiKey_RightArrow)) && !LI.auto_play && LI.frame < LI.max_frame) {
      LI.update_frame = true;
      ++LI.frame;
    }

    ImGui::PopButtonRepeat();
    ImGui::SameLine(0.0f, spacing);
    if (ImGui::SliderInt("Frame", &LI.frame, 0, LI.max_frame, "%d"))
      LI.update_frame = true;

    ImGui::SameLine();
    ImGui::Text(":%d", LI.frame);

    /*----------FPS Control----------*/
    ImGui::Text("FPS Control:");
    ImGui::SameLine();
    ImGui::PushButtonRepeat(true);

    if (ImGui::ArrowButton("fps_left", ImGuiDir_Left) && LI.fps > 1)
      --LI.fps;

    ImGui::SameLine(0.0f, spacing);
    if (ImGui::ArrowButton("fps_right", ImGuiDir_Right) && LI.fps < 200)
      ++LI.fps;

    ImGui::SameLine(0.0f, spacing);
    ImGui::SliderInt("FPS", &LI.fps, 1, 200, "%d");

    ImGui::PopButtonRepeat();
    ImGui::SameLine();
    ImGui::Text(":%d", LI.fps);

    /*----------Auto play and Replay----------*/
    ImGui::Checkbox("Auto Play", &LI.auto_play);
    if (LI.auto_play && LI.update_frame && LI.frame < LI.max_frame) {
      LI.update_frame = true;
      ++LI.frame;
    }

    ImGui::SameLine();

    ImGui::Checkbox("Replay", &LI.replay);
    if (LI.replay && LI.update_frame && LI.frame >= LI.max_frame) {
      LI.update_frame = true;
      LI.frame = 0;
    }

    /*----------Save Label Control----------*/
    ImGui::Checkbox("Enable Enter Key for Saving File", &LI.enable_enter_save);
    if (ImGui::Button("Save Label") ||
        ((ImGui::IsKeyDown(ImGuiKey_LeftCtrl) || ImGui::IsKeyDown(ImGuiKey_RightCtrl)) && ImGui::IsKeyDown(ImGuiKey_S)) ||
        ((ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter)) && LI.enable_enter_save)) {
      //

      if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - LI.current_save_time).count() > 100) {
        LI.current_save_time = std::chrono::system_clock::now();
        LI.save_label = true;
        LI.current_save_frame = LI.frame;
      }
    }

    if (LI.current_save_frame != -1) {
      ImGui::SameLine();
      ImGui::Text("Save Label data from Frame: %d", LI.current_save_frame);
    }

    /*----------Output txt file Control----------*/

    if (ImGui::Button("Output File")) {
      static const std::string feature_output_path = filepath + "/include/dataset/test_feature_data.txt";
      static const std::string label_output_path = filepath + "/include/dataset/test_label_data.txt";

      File_handler::output_feature_data(feature_file, feature_output_path);
      File_handler::output_label_data(label_file, label_output_path);
    }

    /*----------Show Label Rect and Auto Label----------*/

    ImGui::Checkbox("Show Label Rect", &LI.show_rect);

    ImGui::SameLine();
    ImGui::Checkbox("Show Nearest Segment", &LI.show_nearest);

    ImGui::SameLine();
    ImGui::Checkbox("Auto Label", &LI.auto_label);

    /*----------------------------------------*/

    ImGui::TreePop();
  }
}

void ShowLabel()
{
  static const std::string filepath = File_handler::get_filepath();
  static const std::string xy_data_path = filepath + "/include/dataset/ball_xy_data.txt";
  static const std::string xy_bin_data_path = filepath + "/include/dataset/ball_xy_data_bin.txt";
  {
    static bool initialize = true;
    if (initialize) {
      LI.max_frame = File_handler::transform_frame(xy_data_path, xy_bin_data_path);
      initialize = false;
    }
  }

  static std::ifstream xy_bin_data_file(xy_bin_data_path, std::ios::in | std::ios::binary);
  if (xy_bin_data_file.fail()) {
    std::cerr << "cant open " << xy_bin_data_path << '\n';
    std::cin.get();
    exit(1);
  }

  namespace fs = std::filesystem;

  static const std::string feature_file_path = filepath + "/include/dataset/test_feature_bin_data.txt";
  static const std::string label_file_path = filepath + "/include/dataset/test_label_bin_data.txt";
  if (!fs::exists(feature_file_path)) std::ofstream create_file(feature_file_path);    // just for creating file.
  if (!fs::exists(label_file_path)) std::ofstream create_file(label_file_path);    // just for creating file.

  static std::fstream feature_file(feature_file_path, std::ios::in | std::ios::out | std::ios::binary);
  if (feature_file.fail()) {
    std::cerr << "cant open " << feature_file_path << '\n';
    std::cin.get();
    exit(1);
  }
  // feature_file.seekg(0, std::ios::beg);

  static std::fstream label_file(label_file_path, std::ios::in | std::ios::out | std::ios::binary);
  if (label_file.fail()) {
    std::cerr << "cant open " << label_file_path << '\n';
    std::cin.get();
    exit(1);
  }
  // label_file.seekg(0, std::ios::beg);

  ImGui::SetNextWindowPos(ImVec2(50, 50), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(500, 500), ImGuiCond_FirstUseEver);
  ImGui::Begin("Label Window");

  if (LI.auto_play) {
    // check the fps and determine if it should update frame
    if (auto now = std::chrono::system_clock::now();
        std::chrono::duration_cast<std::chrono::milliseconds>(now - LI.current_time).count() > (1000 / LI.fps)) {
      LI.current_time = now;
      LI.update_frame = true;
    }
    else {
      LI.update_frame = false;
    }
  }

  ShowLabelInformation(feature_file, label_file);

  static Eigen::MatrixXd feature_matrix;
  static std::vector<Eigen::MatrixXd> segment_vec;
  static std::vector<int> segment_label;

  static std::vector<int> label_size_vec(LI.max_frame + 1);
  static std::vector<int> label_index_vec(LI.max_frame + 1);
  static std::vector<int> feature_size_vec(LI.max_frame + 1);
  static std::vector<int> feature_index_vec(LI.max_frame + 1);

  if (LI.clean_data) {
    const std::string feature_output_path = filepath + "/include/dataset/test_feature_data.txt";
    const std::string label_output_path = filepath + "/include/dataset/test_label_data.txt";

    namespace fs = std::filesystem;

    if (fs::exists(feature_file_path)) {
      feature_file.seekg(0, std::ios::beg);
      feature_file.seekp(0, std::ios::beg);
      fs::resize_file(feature_file_path, 0);
      std::fill(feature_size_vec.begin(), feature_size_vec.end(), 0);
      std::fill(feature_index_vec.begin(), feature_index_vec.end(), 0);
    }
    if (fs::exists(label_file_path)) {
      label_file.seekg(0, std::ios::beg);
      label_file.seekp(0, std::ios::beg);
      fs::resize_file(label_file_path, 0);
      std::fill(label_size_vec.begin(), label_size_vec.end(), 0);
      std::fill(label_index_vec.begin(), label_index_vec.end(), 0);
    }
    if (fs::exists(feature_output_path)) fs::resize_file(feature_output_path, 0);
    if (fs::exists(label_output_path)) fs::resize_file(label_output_path, 0);

    LI.clean_data = false;
  }

  if (LI.update_frame) {
    LI.update_frame = false;

    File_handler::read_frame(xy_bin_data_file, LI.xy_data, LI.frame);
    std::tie(feature_matrix, segment_vec) = section_to_feature(LI.xy_data);

    segment_label.clear();
    segment_label.resize(segment_vec.size());
    segment_label.shrink_to_fit();

    if (label_size_vec[LI.frame] != 0) {
      label_file.seekg(label_index_vec[LI.frame], std::ios::beg);
      label_file.read(reinterpret_cast<char *>(segment_label.data()), label_size_vec[LI.frame]);
    }
  }

  if (ImGui::TreeNodeEx("Label window")) {
    if (ImPlot::BeginPlot("Label", ImVec2(static_cast<float>(LI.window_size), static_cast<float>(LI.window_size)))) {
      ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 1);
      ImPlot::SetupAxes("x", "y");
      ImPlot::SetupAxisLimits(ImAxis_X1, -5.0, 5.0);
      ImPlot::SetupAxisLimits(ImAxis_Y1, -10, 10);

      double order_using_xy = 0;
      ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 0, ImVec4(0, 0.7f, 0, 1), IMPLOT_AUTO, ImVec4(0, 0.7f, 0, 1));
      ImPlot::PlotScatter("Normal Point", &order_using_xy, &order_using_xy, 1);
      ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 0, ImVec4(1, 0, 0, 1), IMPLOT_AUTO, ImVec4(1, 0, 0, 1));
      ImPlot::PlotScatter("Ball Section", &order_using_xy, &order_using_xy, 1);

      static ImPlotRect rect(-1, 1, -1, 1);
      if (LI.show_rect)
        ImPlot::DragRect(0, &rect.X.Min, &rect.Y.Min, &rect.X.Max, &rect.Y.Max, ImVec4(1, 0, 1, 1), ImPlotDragToolFlags_None);

      int nearest_index = -1;
      double nearest_dis = -1;
      double nearest_x = 0, nearest_y = 0;

      for (int i = 0; i < segment_vec.size(); ++i) {
        const Eigen::MatrixXd &segment = segment_vec[i];

        Eigen::ArrayXd segment_x = segment.col(0).array();
        Eigen::ArrayXd segment_y = segment.col(1).array();
        double X_mean = segment_x.mean();
        double Y_mean = segment_y.mean();
        int segment_size = segment_x.size();

        if (ImPlot::IsPlotHovered() && ImGui::IsMouseClicked(0)) {
          ImPlotPoint click_point = ImPlot::GetPlotMousePos();

          for (auto data_point : segment.rowwise()) {
            if ((data_point(0) - LI.label_mouse_area < click_point.x) && (click_point.x < data_point(0) + LI.label_mouse_area) && (data_point(1) - LI.label_mouse_area < click_point.y) && (click_point.y < data_point(1) + LI.label_mouse_area)) {
              if (segment_label[i] == 1)
                segment_label[i] = 0;
              else
                segment_label[i] = 1;

              break;
            }
          }
        }

        if (LI.show_rect && LI.auto_label) {
          for (auto data_point : segment.rowwise()) {
            if ((rect.X.Min < data_point(0) && data_point(0) < rect.X.Max) && (rect.Y.Min < data_point(1) && data_point(1) < rect.Y.Max)) {
              segment_label[i] = 1;

              break;
            }
          }
        }

        if (LI.show_nearest) {
          double point_dis = std::sqrt(std::pow(X_mean, 2) + std::pow(Y_mean, 2));
          if (point_dis < nearest_dis || nearest_dis == -1) {
            nearest_dis = point_dis;
            nearest_index = i;
            nearest_x = X_mean;
            nearest_y = Y_mean;
          }
        }

        if (segment_label[i] == 1) {
          ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 1, ImVec4(1, 0, 0, 1), IMPLOT_AUTO, ImVec4(1, 0, 0, 1));
          ImPlot::PlotScatter("Ball Section", segment_x.data(), segment_y.data(), segment_size);
        }
        else {
          ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 1, color_arr[i % 9], IMPLOT_AUTO, color_arr[i % 9]);
          ImPlot::PlotScatter("Normal Point", segment_x.data(), segment_y.data(), segment_size);
        }
      }

      if (LI.show_nearest) {
        if (LI.auto_label) {
          segment_label[nearest_index] = 1;
          Eigen::ArrayXd segment_x = segment_vec[nearest_index].col(0).array();
          Eigen::ArrayXd segment_y = segment_vec[nearest_index].col(1).array();
          int segment_size = segment_x.size();
          ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 1, ImVec4(1, 0, 0, 1), IMPLOT_AUTO, ImVec4(1, 0, 0, 1));
          ImPlot::PlotScatter("Ball Section", segment_x.data(), segment_y.data(), segment_size);
        }

        ImPlot::SetNextLineStyle(ImVec4(1, 0, 0, 1));
        ImPlot::PlotLine("nearest_line", &nearest_x, &nearest_y, 2, 0, 0, sizeof(ImPlotPoint));
      }


      if (bool has_been_writed = (label_size_vec[LI.frame] == 0);
          has_been_writed &&
          (LI.save_label || (LI.show_rect && LI.auto_label) || (LI.show_nearest && LI.auto_label))) {
        LI.save_label = false;
        ++LI.writed_frame_numbers;

        // set the size of the frame
        feature_size_vec[LI.frame] = feature_matrix.size() * sizeof(double);
        label_size_vec[LI.frame] = segment_vec.size() * sizeof(int);

        int feature_index = 0, label_index = 0;
        // calculate the index of this feature in the binary file
        for (int sec_i = 0; sec_i < LI.frame; ++sec_i) {
          feature_index += feature_size_vec[sec_i];    // if the frame didn't be writed, the size will be zero
          label_index += label_size_vec[sec_i];    // calculate the index of the label in the binary file
        }

        feature_index_vec[LI.frame] = feature_index;    // It means the nth frame will been writed at `feature_index`
        label_index_vec[LI.frame] = label_index;    // It means the nth label will been writed at `label_index`

        // upload the max frame has been writed
        if (LI.frame > LI.writed_max_frame)
          LI.writed_max_frame = LI.frame;

        // if it's insert, not append, then move all the data after this frame one chunk
        if (LI.frame < LI.writed_max_frame && has_been_writed) {
          static const std::string tmp_feature_filepath = filepath + "/include/dataset/tmp_feature_buffer_data";
          static const std::string tmp_label_filepath = filepath + "/include/dataset/tmp_lable_buffer_data";

          std::fstream file_feature_buf(tmp_feature_filepath, std::ios::in | std::ios::out | std::ios::binary | std::ios::trunc);
          if (file_feature_buf.fail()) {
            std::cerr << "Cannot open file" << tmp_feature_filepath << '\n';
            std::cin.get();
            exit(1);
          }

          std::fstream file_label_buf(tmp_label_filepath, std::ios::in | std::ios::out | std::ios::binary | std::ios::trunc);
          if (file_label_buf.fail()) {
            std::cerr << "Cannot open file" << tmp_label_filepath << '\n';
            std::cin.get();
            exit(1);
          }

          std::vector<double> section_feature_buf;
          std::vector<int> section_label_buf;

          // copy all the data after this frame and move it
          feature_file.seekg(feature_index_vec[LI.frame], std::ios::beg);    // copy from the position will begin written
          label_file.seekg(label_index_vec[LI.frame], std::ios::beg);

          for (int sec_i = LI.frame + 1; sec_i <= LI.writed_max_frame; ++sec_i) {
            if (feature_size_vec[sec_i] != 0) {
              // copy the section(one frame) data
              section_feature_buf.resize(feature_size_vec[sec_i] / sizeof(double));
              section_feature_buf.shrink_to_fit();
              feature_file.read(reinterpret_cast<char *>(section_feature_buf.data()), feature_size_vec[sec_i]);    // copy
              file_feature_buf.write(reinterpret_cast<char *>(section_feature_buf.data()), feature_size_vec[sec_i]);    // and write

              section_label_buf.resize(label_size_vec[sec_i] / sizeof(int));
              section_label_buf.shrink_to_fit();
              label_file.read(reinterpret_cast<char *>(section_label_buf.data()), label_size_vec[sec_i]);    // copy
              file_label_buf.write(reinterpret_cast<char *>(section_label_buf.data()), label_size_vec[sec_i]);    // and write

              feature_index_vec[sec_i] += feature_size_vec[LI.frame];    // add the offset of the data we will insert
              label_index_vec[sec_i] += label_size_vec[LI.frame];    // add the offset of the data we will insert
            }
          }

          file_feature_buf.flush();
          file_label_buf.flush();
          file_feature_buf.seekg(0, std::ios::beg);
          file_label_buf.seekg(0, std::ios::beg);

          File_handler::write_bin_feature_data(feature_file, feature_index, feature_matrix);
          File_handler::write_bin_label_data(label_file, label_index, segment_label);

          for (int sec_i = LI.frame + 1; sec_i <= LI.writed_max_frame; ++sec_i) {
            if (feature_size_vec[sec_i] != 0) {
              section_feature_buf.resize(feature_size_vec[sec_i] / sizeof(double));
              section_feature_buf.shrink_to_fit();
              file_feature_buf.read(reinterpret_cast<char *>(section_feature_buf.data()), feature_size_vec[sec_i]);    // copy
              feature_file.write(reinterpret_cast<char *>(section_feature_buf.data()), feature_size_vec[sec_i]);    // and write

              section_label_buf.resize(label_size_vec[sec_i] / sizeof(int));
              section_label_buf.shrink_to_fit();
              file_label_buf.read(reinterpret_cast<char *>(section_label_buf.data()), label_size_vec[sec_i]);    // copy
              label_file.write(reinterpret_cast<char *>(section_label_buf.data()), label_size_vec[sec_i]);    // and write
            }
          }

          feature_file.flush();
          label_file.flush();
        }
        else {
          File_handler::write_bin_feature_data(feature_file, feature_index, feature_matrix);
          File_handler::write_bin_label_data(label_file, label_index, segment_label);
        }
      }

      ImPlot::PopStyleVar();
      ImPlot::EndPlot();
    }

    ImGui::TreePop();
  }

  ImGui::End();
}