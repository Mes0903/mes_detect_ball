#include "adaboost_classifier.h"
#include "confusion.h"
#include "load_matrix.h"
#include "make_feature.h"
#include "weak_learner.h"
#include "segment.h"
#include "normalize.h"
#include <iostream>
#include <Eigen/Dense>
#include <limits>

int main()
{
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
  Eigen::MatrixXd train_X = readDataSet("/home/mes/catkin_ws/src/mes_detect_ball/include/dataset/box_train_xn.txt", 2888, 5); // file, row, col

  puts("reading testing data...");
  Eigen::MatrixXd test_X = readDataSet("/home/mes/catkin_ws/src/mes_detect_ball/include/dataset/box_test_xn.txt", 1238, 5);

  puts("reading training label...");
  Eigen::VectorXd train_Y = readLabel("/home/mes/catkin_ws/src/mes_detect_ball/include/dataset/box_train_yn.txt", 2888); // file, segment num(row)

  puts("reading testing label...");
  Eigen::VectorXd test_Y = readLabel("/home/mes/catkin_ws/src/mes_detect_ball/include/dataset/box_test_yn.txt", 1238);

  Normalizer normalizer;

  if (case_num == 1)
  {
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
      Eigen::MatrixXd confusion = cal_confusion_matrix(test_Y, pred_Y);
      std::cout << confusion << '\n';

      A.store_weight("/home/mes/catkin_ws/src/mes_detect_ball/include/weight_data/adaboost_box_weight.txt", confusion(0, 0), confusion(1, 0), normalizer);
    }
  }
  else
  {
    Adaboost A;
    A.load_weight("/home/mes/catkin_ws/src/mes_detect_ball/include/weight_data/adaboost_box_weight.txt", normalizer);

    train_X = normalizer.transform(train_X);
    test_X = normalizer.transform(test_X);
    puts("make prediction");
    Eigen::VectorXd pred_Y = A.predict(test_X);
    puts("cal confusion matrix");
    Eigen::MatrixXd confusion = cal_confusion_matrix(test_Y, pred_Y);
    std::cout << confusion << '\n';
  }
}