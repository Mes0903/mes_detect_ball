/**
 * @file training_ball.cpp
 * @author Mes
 * @brief Traning the Adaboost to classified if an object is an ball, then stored the weighting.
 * @version 0.1
 * @date 2022-11-17
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "adaboost_classifier.h"
#include "make_feature.h"
#include "weak_learner.h"
#include "segment.h"
#include "normalize.h"
#include "file_handler.h"

#include <iostream>
#include <Eigen/Dense>
#include <limits>

int main([[maybe_unused]] int argc, char **argv)
{
  const std::string filepath = get_filepath(argv[0]);
  int case_num = 0;
  int sample = 0;

  std::cout << "Input 1 if training, others if loading\n>";
  std::cin >> case_num;

  if (case_num == 1)
  {
    std::cout << "input sample numbers\n>";
    std::cin >> sample;
  }
  std::cin.clear();
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  /* fitting */
  puts("read training data...");
  Eigen::MatrixXd train_X = Load_Matrix::readDataSet(filepath + "/include/dataset/ball_train_x.txt", 2248, 5); // file, row, col

  puts("reading testing data...");
  Eigen::MatrixXd test_X = Load_Matrix::readDataSet(filepath + "/include/dataset/ball_test_x.txt", 2230, 5);

  puts("reading training label...");
  Eigen::VectorXd train_Y = Load_Matrix::readLabel(filepath + "/include/dataset/ball_train_y.txt", 2248); // file, segment num(row)

  puts("reading testing label...");
  Eigen::VectorXd test_Y = Load_Matrix::readLabel(filepath + "/include/dataset/ball_test_y.txt", 2230);

  if (case_num == 1)
  {
    Normalizer normalizer;
    normalizer.fit(train_X);
    train_X = normalizer.transform(train_X);
    test_X = normalizer.transform(test_X);

    for (int i = 0; i < sample; ++i)
    {
      puts("start tranning");
      Adaboost A(100);
      A.fit(train_X, train_Y);

      // prediction
      puts("make prediction");
      Eigen::VectorXd pred_Y = A.predict(test_X);

      puts("cal confusion matrix");
      Eigen::MatrixXd confusion = A.cal_confusion_matrix(test_Y, pred_Y);
      std::cout << confusion << '\n';

      Weight_handle::store_weight(confusion, filepath + "/include/weight_data/adaboost_ball_weight.txt", A, normalizer);
    }
  }
  else
  {
    Normalizer normalizer;
    Adaboost A;
    Weight_handle::load_weight(filepath + "/include/weight_data/adaboost_ball_weight.txt", A, normalizer);

    test_X = normalizer.transform(test_X);
    puts("make prediction");
    Eigen::VectorXd pred_Y = A.predict(test_X);
    puts("cal confusion matrix");
    Eigen::MatrixXd confusion = A.cal_confusion_matrix(test_Y, pred_Y);
    std::cout << confusion << '\n';
  }
}