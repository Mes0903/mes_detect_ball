/**
 * @file file_handler.cpp
 * @author Mes (mes900903@gmail.com) (Discord: Mes#0903)
 * @brief Load the file and store weights to file, any functions related to file operations should be here.
 * @version 0.1
 * @date 2022-12-15
 */

#include "file_handler.h"
#include "normalize.h"
#include "make_feature.h"

#include <Eigen/Eigen>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unordered_set>

#if __cplusplus >= 202002L
#include <string_view>
#include <ranges>
#else
#include <type_traits>
#endif

#define CLEAN_STREAM \
  stream.str("");    \
  stream.clear()

namespace Load_Matrix {
  /**
   * @brief Read the data from the filepath to the matrix.
   *
   * @param filepath The file which would be loaded to matrix.
   * @param ROWS The lines number of the file.
   * @param COLS The cols number of the file.
   * @return Eigen::MatrixXd The matrix which have completed loading.
   */
  Eigen::MatrixXd readDataSet(const std::string filepath, const int ROWS, const int COLS)
  {
    std::ifstream infile(filepath);
    if (infile.fail()) {
      std::cerr << "cant found " << filepath << '\n';
      std::cin.get();
      exit(1);
    }

    Eigen::MatrixXd result(ROWS, COLS);
    std::string line;
    std::stringstream stream;
    int row = 0;
    for (int cnt = 0; cnt < ROWS; ++cnt) {
      double buff;
      getline(infile, line);    // read every line of the file
      stream << line;
      for (int col = 0; col < COLS; ++col) {
        stream >> buff;
        result(row, col) = buff;
      }
      ++row;

      CLEAN_STREAM;
    }

    infile.close();

    return result;
  };

  /**
   * @brief Read the Labeling data from the file and load it to the matrix
   *
   * @param filepath The file which would be loaded to matrix.
   * @param SIZE The lines number of the file.
   * @return Eigen::VectorXd  The vector which have completed loading.
   */
  Eigen::VectorXd readLabel(const std::string filepath, const int SIZE)
  {
    std::ifstream infile(filepath);
    if (infile.fail()) {
      std::cerr << "cant found " << filepath << '\n';
      std::cin.get();
      exit(1);
    }

    Eigen::VectorXd result(SIZE);
    std::string line;
    std::stringstream stream;

    int row = 0;
    for (int cnt = 0; cnt < SIZE; ++cnt) {
      double buff;
      getline(infile, line);    // read every line of the file
      stream << line;
      stream >> buff;
      result(row) = buff;
      ++row;

      CLEAN_STREAM;
    }

    infile.close();
    return result;
  }
}    // namespace Load_Matrix


namespace File_handler {

  /**
   * @brief Return the project directory path.
   *
   * @param filepath the executable file path, which is argv[0].
   * @return std::string The project directory path
   */
  std::string get_filepath([[maybe_unused]] const char *filepath)
  {
#if _WIN32
    return "D:/document/GitHub/mes_detect_ball";
#else
    std::string buf = filepath;
    std::string path;

#if __cplusplus >= 202002L
    for (const std::string token : buf | std::ranges::views::split('/')    // split the file path by '/'
                                     | std::ranges::views::transform(    // transform the output type to the std::string
                                         [](auto &&rng) { return std::string(&*rng.begin(), std::ranges::distance(rng)); })) {
      if (token != "") {
        path += "/" + token;

        if (token == "catkin_ws") {
          path += "/src/mes_detect_ball";
          break;
        }
      }
    }
#else
    std::string delimiter = "/";

    std::size_t pos = 0;
    std::string token;
    while ((pos = buf.find(delimiter)) != std::string::npos) {
      token = buf.substr(0, pos);
      if (token != "") {
        path += '/' + token;

        if (token == "catkin_ws") {
          path += "/src/mes_detect_ball";
          break;
        }
      }

      buf.erase(0, pos + delimiter.length());
    }

#endif

    return path;

#endif    // __linux__
  }

  int transform_frame(const std::string &in_filepath, const std::string &out_filepath)
  {
    int max_frame = 0;

    std::ifstream infile(in_filepath);
    if (infile.fail()) {
      std::cerr << "cant found " << in_filepath << '\n';
      std::cin.get();
      exit(1);
    }

    std::ofstream outfile(out_filepath, std::ios::binary);
    if (outfile.fail()) {
      std::cerr << "cant found " << out_filepath << '\n';
      std::cin.get();
      exit(1);
    }

    std::string line;
    std::stringstream ss;
    double x, y;
    while (std::getline(infile, line)) {
      ++max_frame;

      ss << line;
      ss >> x >> y;

      outfile.write(reinterpret_cast<char *>(&x), sizeof(double));
      outfile.write(reinterpret_cast<char *>(&y), sizeof(double));

      ss.str("");
      ss.clear();
    }

    max_frame /= 720;
    --max_frame;

    return max_frame;
  }

  void read_frame(std::ifstream &infile, Eigen::MatrixXd &xy_data, const int frame)
  {
    infile.seekg(frame * sizeof(double) * 2 * 720, std::ios::beg);

    for (int i{}; i < 720; ++i) {
      infile.read(reinterpret_cast<char *>(&xy_data(i, 0)), sizeof(double));
      infile.read(reinterpret_cast<char *>(&xy_data(i, 1)), sizeof(double));
    }
  }

  void write_bin_feature_data(std::fstream &feature_file, const int feature_index, const Eigen::MatrixXd &feature_matrix)
  {
    feature_file.seekp(feature_index, std::ios::beg);
    for (const auto &row : feature_matrix.rowwise()) {
      for (double feature : row)
        feature_file.write(reinterpret_cast<char *>(&feature), sizeof(double));
    }
  }

  void write_bin_label_data(std::fstream &label_file, const int label_index, const std::vector<int> &segment_label)
  {
    label_file.seekp(label_index, std::ios::beg);
    for (int label : segment_label)
      label_file.write(reinterpret_cast<char *>(&label), sizeof(int));
  }

  void output_feature_data(std::fstream &feature_file, const std::string &filepath)
  {
    auto bk_p = feature_file.tellg();

    std::ofstream outfile(filepath, std::ios::out | std::ios::trunc);
    if (outfile.fail()) {
      std::cerr << "Cannot open file" << filepath << '\n';
      std::cin.get();
      return;
    }

    feature_file.seekg(0, std::ios::beg);
    double buf[FEATURE_NUM];
    while (feature_file.read(reinterpret_cast<char *>(buf), FEATURE_NUM * sizeof(double))) {
      for (int i{}; i < FEATURE_NUM; ++i)
        outfile << buf[i] << " \n"[i == FEATURE_NUM - 1];
    }

    feature_file.seekg(bk_p, std::ios::beg);
    feature_file.clear();
  }

  void output_label_data(std::fstream &label_file, const std::string &filepath)
  {
    auto bk_p = label_file.tellg();

    std::ofstream outfile(filepath, std::ios::out | std::ios::trunc);
    if (outfile.fail()) {
      std::cerr << "Cannot open file" << filepath << '\n';
      std::cin.get();
      return;
    }

    label_file.seekg(0, std::ios::beg);
    int buf;
    while (label_file.read(reinterpret_cast<char *>(&buf), sizeof(int)))
      outfile << buf << '\n';

    label_file.seekg(bk_p, std::ios::beg);
    label_file.clear();
  }

}    // namespace File_handler