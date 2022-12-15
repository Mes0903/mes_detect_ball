/**
 * @file file_handler.cpp
 * @author Mes (mes900903@gmail.com) (Discord: Mes#0903)
 * @brief Load the file and store weights to file, any functions related to file operations should be here.
 * @version 0.1
 * @date 2022-12-15
 */

#include "file_handler.h"
#include "normalize.h"
#include <Eigen/Eigen>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>

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
      std::cout << "cant found " << filepath << '\n';
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
      std::cout << "cant found " << filepath << '\n';
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
}    // namespace File_handler