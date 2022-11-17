#pragma once
#ifndef __FILE_HANDLER_H__
#define __FILE_HANDLER_H__

/**
 * @file file_handler.h
 * @author Mes
 * @brief Handling the file operations, contained
 *        1. get file path
 *        2. load matrix from file, for data and label using
 *        3. store weighting for Adaboost and Normalizer
 *
 * @version 0.1
 * @date 2022-11-17
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <string>
#include <ranges>
#include <string_view>
#include <Eigen/Eigen>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>

#define CLEAN_STREAM \
  stream.str("");    \
  stream.clear()

/**
 * @brief Return the project directory path.
 *
 * @param filepath the executable file path, which is argv[0].
 * @return std::string The project directory path
 */
std::string get_filepath(const char *filepath)
{
  std::string buf = filepath;
  std::string path;

  for (const std::string word : buf | std::ranges::views::split('/') // split the file path by '/'
                                    | std::ranges::views::transform( // transform the output type to the std::string
                                          [](auto &&rng)
                                          { return std::string(&*rng.begin(), std::ranges::distance(rng)); }))
  {
    if (word != "")
    {
      path += "/" + word;

      if (word == "catkin_ws")
      {
        path += "/src/mes_detect_ball";
        break;
      }
    }
  }

  return path;
}

namespace Load_Matrix
{
  /**
   * @brief Transforming the matrix from [theta, r] data to [x, y] data.
   *
   * @param data The [thera, r] matrix.
   * @param ROWS The rows number of the matrix.
   */
  void transform_to_xy(Eigen::MatrixXd &data, const uint32_t ROWS)
  {
    for (uint32_t i = 0; i < ROWS; i++)
    {
      const double theta = M_PI * data(i, 0) / 180; // transform the radian to angle
      const double r = data(i, 1);                  // radius

      data(i, 0) = r * std::cos(theta); // x
      data(i, 1) = r * std::sin(theta); // y
    }
  }

  /**
   * @brief Read the data from the filepath to the matrix.
   *
   * @param filepath The file which would be loaded to matrix.
   * @param ROWS The lines number of the file.
   * @param COLS The cols number of the file.
   * @return Eigen::MatrixXd The matrix which have completed loading.
   */
  Eigen::MatrixXd readDataSet(const std::string filepath, const uint32_t ROWS, const uint32_t COLS)
  {
    std::ifstream infile(filepath);
    if (infile.fail())
    {
      std::cout << "cant found " << filepath << '\n';
      exit(1);
    }

    Eigen::MatrixXd result(ROWS, COLS);
    std::string line;
    std::stringstream stream;
    uint32_t row = 0;
    for (uint32_t cnt = 0; cnt < ROWS; ++cnt)
    {
      double buff;
      getline(infile, line); // read every line of the file
      stream << line;
      for (uint32_t col = 0; col < COLS; ++col)
      {
        stream >> buff;
        result(row, col) = buff;
      }
      ++row;

      CLEAN_STREAM;
    }

    infile.close();

    puts("transforming data to xy data");
    transform_to_xy(result, ROWS);
    return result;
  };

  /**
   * @brief Read the Labeling data from the file and load it to the matrix
   *
   * @param filepath The file which would be loaded to matrix.
   * @param SIZE The lines number of the file.
   * @return Eigen::VectorXd  The vector which have completed loading.
   */
  Eigen::VectorXd readLabel(const std::string filepath, const uint32_t SIZE)
  {
    std::ifstream infile(filepath);
    if (infile.fail())
    {
      std::cout << "cant found " << filepath << '\n';
      exit(1);
    }

    Eigen::VectorXd result(SIZE);
    std::string line;
    std::stringstream stream;

    uint32_t row = 0;
    for (uint32_t cnt = 0; cnt < SIZE; ++cnt)
    {
      double buff;
      getline(infile, line); // read every line of the file
      stream << line;
      stream >> buff;
      result(row) = buff;
      ++row;

      CLEAN_STREAM;
    }

    infile.close();
    return result;
  }
}

namespace Weight_handle
{

  namespace detail
  {
    /**
     * @brief Check if the weight in class can be stored.
     * @param ins The instance of the class.
     */
    template <typename T>
    concept can_store = requires(T &ins, const std::string filepath, std::ofstream &outfile)
    {
      ins.store_weight(filepath, outfile); // check if the class have function `store_weight(filepath, outfile)`
    };

    /**
     * @brief Check if the weight in class can be loaded.
     * @param ins The instance of the class.
     */
    template <typename T>
    concept can_load = requires(T &ins, const std::string filepath, std::ifstream &infile)
    {
      ins.load_weight(filepath, infile); // check if the class have function `load_weight(filepath, infile)`
    };

    /**
     * @brief The implementation for loading data.
     *
     * @param filepath The file which would be writed.
     * @param ins The class instance.
     * @param first The head of parameter pack, a class instance.
     * @param instances The parameter pack, class instances.
     */
    template <typename T>
    void __store_weight(const std::string filepath, std::ofstream &outfile, T &ins) requires can_store<T>
    {
      ins.store_weight(filepath, outfile); // call the function the instance define
    }

    template <typename T, typename... A>
    void __store_weight(const std::string filepath, std::ofstream &outfile, T &first, A &...instances)
    {
      __store_weight(filepath, outfile, first);        // call the store_weight function of the first instance
      __store_weight(filepath, outfile, instances...); // recursive call the store_weight function
    }

    /**
     * @brief The implementation for storing data.
     *
     * @param filepath The file which would be stored.
     * @param ins The class instance.
     * @param first The head of parameter pack, a class instance.
     * @param instances The parameter pack, class instances.
     */
    template <typename T>
    void __load_weight(const std::string filepath, std::ifstream &infile, T &ins) requires can_load<T>
    {
      ins.load_weight(filepath, infile); // call the function the instance define
    }

    template <typename T, typename... A>
    void __load_weight(const std::string filepath, std::ifstream &infile, T &first, A &...instances)
    {
      __load_weight(filepath, infile, first);        // call the load_weight function of the first instance
      __load_weight(filepath, infile, instances...); // recursive call the load_weight function
    }
  } // namespace Weight_handle::detail

  /**
   * @brief The user interface for storing data.
   *
   * @param filepath The file which would be stored.
   * @param instances The parameter pack, class instances.
   */
  template <typename... T>
  void store_weight(const Eigen::MatrixXd &confusion, const std::string filepath, T &...instances)
  {

    std::ifstream infile(filepath);
    if (infile.fail())
    {
      std::cerr << "cant read " << filepath << '\n';
      exit(1);
    }

    double correct_rate;                             // the correct rate before
    uint32_t store_TN, store_TP, store_FN, store_FP; // the TP, TP, FN, FP stored before

    std::string line;
    std::stringstream stream;
    getline(infile, line);
    stream << line;
    stream >> correct_rate >> store_TN >> store_TP >> store_FN >> store_FP;
    CLEAN_STREAM;
    infile.close();

    double store_TP_FN = static_cast<double>(store_TP) / (store_TP + store_FN); // TP/FN
    double store_TN_FP = static_cast<double>(store_TN) / (store_TN + store_FP); // TN/FP

    uint32_t TP = confusion(0, 0), FP = confusion(0, 1), FN = confusion(1, 0), TN = confusion(1, 1);
    double current_TP_FN = static_cast<double>(TP) / (TP + FN);
    double current_TN_FP = static_cast<double>(TN) / (TN + FP);

    std::cout << "calculated accuracy: " << current_TP_FN << "\nbest accuract before: " << store_TP_FN << '\n';

    // compact the correct rate between storing file and current training data
    if (current_TP_FN < store_TP_FN)
    {
      std::cout << "This weight won't be saved since its correct rate is lower than the original one\n";
      return;
    }
    else if (current_TP_FN == store_TP_FN) // if TP/FN is same, compact the TN/FP.
    {
      if (current_TN_FP <= store_TN_FP)
      {
        std::cout << "This weight won't be saved since its correct rate is lower than the original one\n";
        return;
      }
    }

    std::ofstream outfile(filepath);
    if (outfile.fail())
    {
      std::cout << "cant found " << filepath << '\n';
      exit(1);
    }

    detail::__store_weight(filepath, outfile, instances...); // call the implementation of store_weight

    outfile.close();
  }

  /**
   * @brief The API for loading data.
   *
   * @param filepath The file which would be stored.
   * @param instances The parameter pack, class instances.
   */
  template <typename... T>
  void load_weight(const std::string filepath, T &...instances)
  {
    std::ifstream infile(filepath);
    if (infile.fail())
    {
      std::cout << "cant found " << filepath << '\n';
      exit(1);
    }

    detail::__load_weight(filepath, infile, instances...); // call the implementation of store_weight

    infile.close();
  }
} // namespace Weight_handle

#endif