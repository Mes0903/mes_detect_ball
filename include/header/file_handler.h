#pragma once
#ifndef __FILE_HANDLER_H__
#define __FILE_HANDLER_H__

#include <string>
#include <ranges>
#include <string_view>

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

  for (auto word : buf | std::ranges::views::split('/')) {
    std::string_view seg_view = { word.begin(), word.end() };
    path += "/" + buf;

    if (seg_view == "mes_detect_ball")
      break;
  }

  return path;
}

namespace Weight_handle {

  /**
   * @brief check if the weight in class can be stored
   * @param ins the instance of the class
   */
  template <typename T>
  concept can_store = requires(T &ins, const std::string filepath)
  {
    ins.store_weight(filepath);
  };

  /**
   * @brief check if the weight in class can be loaded
   * @param ins the instance of the class
   */
  template <typename T>
  concept can_load = requires(T &ins, const std::string filepath)
  {
    ins.load_weight(filepath);
  };

  /**
   * @brief the API for loading data
   *
   * @param filepath the file which would be writed
   * @param ins the class instance
   * @param first the head of parameter pack, a class instance
   * @param instances the parameter pack, class instances
   */
  template <typename T>
  void store_weight(const std::string filepath, T &ins) requires can_store<T>
  {
    ins.store_weight(filepath);
  }

  template <typename T, typename... A>
  void store_weight(const std::string filepath, T &first, A &...instances)
  {
    store_weight(filepath, first);
    store_weight(filepath, instances...);
  }

  /**
   * @brief the API for storing data
   *
   * @param filepath the file which would be stored
   * @param ins the class instance
   * @param first the head of parameter pack, a class instance
   * @param instances the parameter pack, class instances
   */
  template <typename T>
  void load_weight(const std::string filepath, T &ins) requires can_load<T>
  {
    ins.load_weight(filepath);
  }

  template <typename T, typename... A>
  void load_weight(const std::string filepath, T &first, A &...instances)
  {
    load_weight(filepath, first);
    load_weight(filepath, instances...);
  }

}    // namespace Weight_handle


#endif