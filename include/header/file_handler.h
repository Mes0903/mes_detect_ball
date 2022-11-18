#pragma once
#ifndef __FILE_HANDLER_H__
#define __FILE_HANDLER_H__

/**
 * @file file_handler.h
 * @author Mes (mes900903@gmail.com) (Discord: Mes#0903)
 * @brief Load the file and store weights to file, any functions related to file operations should be here.
 * @version 0.1
 * @date 2022-11-18
 */

#include <Eigen/Eigen>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>

#if __cplusplus >= 202002L

#include <string_view>
#include <ranges>

#else

#include <type_traits>

#endif

#define CLEAN_STREAM \
  stream.str("");    \
  stream.clear()

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
} // namespace Load_Matrix

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

#if __cplusplus >= 202002L
  for (const std::string token : buf | std::ranges::views::split('/') // split the file path by '/'
                                     | std::ranges::views::transform( // transform the output type to the std::string
                                           [](auto &&rng)
                                           { return std::string(&*rng.begin(), std::ranges::distance(rng)); }))
  {
    if (token != "")
    {
      path += "/" + token;

      if (token == "catkin_ws")
      {
        path += "/src/mes_detect_ball";
        break;
      }
    }
  }
#else
  std::string delimiter = "/";

  std::size_t pos = 0;
  std::string token;
  while ((pos = buf.find(delimiter)) != std::string::npos)
  {
    token = buf.substr(0, pos);
    if (token != "")
    {
      path += '/' + token;

      if (token == "catkin_ws")
      {
        path += "/src/mes_detect_ball";
        break;
      }
    }

    buf.erase(0, pos + delimiter.length());
  }

#endif

  return path;
}

namespace Weight_handle
{

  namespace detail
  {
#if __cplusplus >= 202002L
    /**
     * @brief check if the weight in class can be stored
     * @param ins the instance of the class
     */
    template <typename T>
    concept can_store = requires(T &ins, const std::string filepath, std::ofstream &outfile)
    {
      ins.store_weight(filepath, outfile);
    };

    /**
     * @brief check if the weight in class can be loaded
     * @param ins the instance of the class
     */
    template <typename T>
    concept can_load = requires(T &ins, const std::string filepath, std::ifstream &infile)
    {
      ins.load_weight(filepath, infile);
    };

#endif

    /**
     * @brief the implementation for loading data
     *
     * @param filepath the file which would be writed
     * @param ins the class instance
     * @param first the head of parameter pack, a class instance
     * @param instances the parameter pack, class instances
     */
    template <typename T>
    void store_weight_impl(const std::string filepath, std::ofstream &outfile, T &ins)
#if __cplusplus >= 202002L
        requires can_store<T> // Check if the instance implemented the `store_weight` method by Detection Idioms(Concept requires)
#endif
    {
#if __cplusplus >= 202002L
      ins.store_weight(filepath, outfile);
#else
      /**
       * @brief Check if the instance implemented the `store_weight` method by Detection Idioms(constexpr if)
       */
      if constexpr (std::is_void_v<decltype(std::declval<T>().store_weight(filepath, outfile))>)
        ins.store_weight(filepath, outfile);
      else
        throw std::logic_error("The class didn't implement the store_weight function, or with non void return type.");
#endif
    }

    template <typename T, typename... A>
    void store_weight_impl(const std::string filepath, std::ofstream &outfile, T &first, A &...instances)
    {
      store_weight_impl(filepath, outfile, first);
      store_weight_impl(filepath, outfile, instances...);
    }

    /**
     * @brief the implementation for storing data
     *
     * @param filepath the file which would be stored
     * @param ins the class instance
     * @param first the head of parameter pack, a class instance
     * @param instances the parameter pack, class instances
     */
    template <typename T>
    void load_weight_impl(const std::string filepath, std::ifstream &infile, T &ins)
#if __cplusplus >= 202002L
        requires can_load<T> // Check if the instance implemented the `load_weight` method by Detection Idioms(Concept requires)
#endif
    {
#if __cplusplus >= 202002L
      ins.load_weight(filepath, infile);
#else
      /**
       * @brief Check if the instance implemented the `load_weight` method by Detection Idioms(constexpr if)
       */
      if constexpr (std::is_void_v<decltype(std::declval<T>().load_weight(filepath, infile))>)
        ins.load_weight(filepath, infile);
      else
        throw std::logic_error("The class didn't implement the store_weight function, or with non void return type.");
#endif
    }

    template <typename T, typename... A>
    void load_weight_impl(const std::string filepath, std::ifstream &infile, T &first, A &...instances)
    {
      load_weight_impl(filepath, infile, first);
      load_weight_impl(filepath, infile, instances...);
    }
  } // namespace detail

  /**
   * @brief the API for storing data
   *
   * @param filepath the file which would be stored
   * @param instances the parameter pack, class instances
   */
  template <typename... T>
  void store_weight(const Eigen::MatrixXd &confusion, const std::string filepath, T &...instances)
  {
    // compact the correct rate between storing file and current training data
    std::ifstream infile(filepath);
    if (infile.fail())
    {
      std::cerr << "cant read " << filepath << '\n';
      exit(1);
    }

    std::string line;
    std::stringstream stream;
    getline(infile, line);

    double old_F1_Score;
    uint32_t store_TN, store_TP, store_FN, store_FP;
    stream << line;
    stream >> old_F1_Score >> store_TN >> store_TP >> store_FN >> store_FP;
    stream.str("");
    stream.clear();
    infile.close();

    // compact the correct rate between storing file and current training data
    uint32_t TP = confusion(0, 0), FP = confusion(0, 1), FN = confusion(1, 0), TN = confusion(1, 1);

    double accuracy = static_cast<double>(TP + TN) / (TP + TN + FP + FN);
    double recall = static_cast<double>(TP) / (TP + FN);
    double precision = static_cast<double>(TP) / (TP + FP);
    double F1_Score = 2 * precision * recall / (precision + recall);

    std::cout << "Calculated Accuracy : " << accuracy << '\n'
              << "Calculated recall : " << recall << '\n'
              << "Calculated precision: " << precision << '\n'
              << "Calculated F1 Score: " << F1_Score << '\n'
              << "Best F1 Score: " << old_F1_Score << '\n';

    // compact F1 Score
    if (F1_Score <= old_F1_Score)
    {
      std::cout << "This weight won't be saved since its F1 Score is not better than the original one\n";
      return;
    }

    std::ofstream outfile(filepath);
    if (outfile.fail())
    {
      std::cout << "cant found " << filepath << '\n';
      exit(1);
    }

    detail::store_weight_impl(filepath, outfile, instances...);

    outfile.close();
  }

  /**
   * @brief the API for loading data
   *
   * @param filepath the file which would be stored
   * @param instances the parameter pack, class instances
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

    detail::load_weight_impl(filepath, infile, instances...);

    infile.close();
  }
} // namespace Weight_handle

#endif