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
  int sample = 0;
  std::cout << "input sample numbers\n>";
  std::cin >> sample;

  /* fitting */
  puts("read training data...");
  Eigen::MatrixXd train_X = readDataSet("/home/hypharos/catkin_ws/src/mes_detect_ball/include/dataset/ball_train_x.txt", 2248, 5); // file, row, col

  puts("reading testing data...");
  Eigen::MatrixXd test_X = readDataSet("/home/hypharos/catkin_ws/src/mes_detect_ball/include/dataset/ball_test_x.txt", 2230, 5);

  puts("reading training label...");
  Eigen::VectorXd train_Y = readLabel("/home/hypharos/catkin_ws/src/mes_detect_ball/include/dataset/ball_train_y.txt", 2248); // file, segment num(row)

  puts("reading testing label...");
  Eigen::VectorXd test_Y = readLabel("/home/hypharos/catkin_ws/src/mes_detect_ball/include/dataset/ball_test_y.txt", 2230);

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
    A.store_weight("/home/hypharos/catkin_ws/src/mes_detect_ball/include/weight_data/adaboost_ball_weight.txt", confusion(0, 0), confusion(1, 0));
  }
}