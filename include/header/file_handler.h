#pragma once
#ifndef __FILE_HANDLER_H__
#define __FILE_HANDLER_H__

#include <string>
#include <ranges>
#include <string_view>
#include <Eigen/Eigen>

/**
 * @brief return the project directory path
 *
 * @param filepath the executable file path, which is argv[0].
 * @return std::string the project directory path
 */
std::string get_filepath(const char *filepath)
{
  std::string buf = filepath;
  std::string path;

  for (auto word : buf | std::ranges::views::split('/') | std::ranges::views::transform([](auto &&rng)
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

namespace Weight_handle
{

  namespace detail
  {
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

    /**
     * @brief the implementation for loading data
     *
     * @param filepath the file which would be writed
     * @param ins the class instance
     * @param first the head of parameter pack, a class instance
     * @param instances the parameter pack, class instances
     */
    template <typename T>
    void __store_weight(const std::string filepath, std::ofstream &outfile, T &ins) requires can_store<T>
    {
      ins.store_weight(filepath, outfile);
    }

    template <typename T, typename... A>
    void __store_weight(const std::string filepath, std::ofstream &outfile, T &first, A &...instances)
    {
      __store_weight(filepath, outfile, first);
      __store_weight(filepath, outfile, instances...);
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
    void __load_weight(const std::string filepath, std::ifstream &infile, T &ins) requires can_load<T>
    {
      ins.load_weight(filepath, infile);
    }

    template <typename T, typename... A>
    void __load_weight(const std::string filepath, std::ifstream &infile, T &first, A &...instances)
    {
      __load_weight(filepath, infile, first);
      __load_weight(filepath, infile, instances...);
    }
  } // namespace Weight_handle::detail

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

    double correct_rate;
    uint32_t store_TN, store_TP, store_FN, store_FP;
    stream << line;
    stream >> correct_rate >> store_TN >> store_TP >> store_FN >> store_FP;
    stream.str("");
    stream.clear();
    infile.close();

    double store_TP_FN = static_cast<double>(store_TP) / (store_TP + store_FN);
    double store_TN_FP = static_cast<double>(store_TN) / (store_TN + store_FP);

    uint32_t TP = confusion(0, 0), FP = confusion(0, 1), FN = confusion(1, 0), TN = confusion(1, 1);
    double current_TP_FN = static_cast<double>(TP) / (TP + FN);
    double current_TN_FP = static_cast<double>(TN) / (TN + FP);

    std::cout << "calculated accuracy: " << current_TP_FN << "\nbest accuract before: " << store_TP_FN << '\n';
    // compact TP / TP + FN
    if (current_TP_FN < store_TP_FN)
    {
      std::cout << "This weight won't be saved since its correct rate is lower than the original one\n";
      return;
    }
    else if (current_TP_FN == store_TP_FN)
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

    detail::__store_weight(filepath, outfile, instances...);

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

    detail::__load_weight(filepath, infile, instances...);

    infile.close();
  }
} // namespace Weight_handle

#endif