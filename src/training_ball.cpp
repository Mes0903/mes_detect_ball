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
  /* fitting */
  puts("read training data...");
  Eigen::MatrixXd train_X = readDataSet("/home/mes/catkin_ws/src/mes_detect_ball/include/dataset/train_ball_X.txt", 2248, 5); // file, row, col

  puts("reading testing data...");
  Eigen::MatrixXd test_X = readDataSet("/home/mes/catkin_ws/src/mes_detect_ball/include/dataset/test_ball_X.txt", 4478, 5);

  puts("reading training label...");
  Eigen::VectorXd train_Y = readLabel("/home/mes/catkin_ws/src/mes_detect_ball/include/dataset/train_ball_Y.txt", 2248); // file, segment num(row)

  puts("reading testing label...");
  Eigen::VectorXd test_Y = readLabel("/home/mes/catkin_ws/src/mes_detect_ball/include/dataset/test_ball_Y.txt", 4478);

  puts("start tranning");
  Adaboost A(100);
  A.fit(train_X, train_Y);

  // prediction
  puts("make prediction");
  Eigen::VectorXd pred_Y = A.predict(test_X);

  puts("cal confusion matrix");
  Eigen::MatrixXd confusion = cal_confusion_matrix(test_Y, pred_Y);

  std::cout << confusion << '\n';
  A.store_weight("/home/mes/catkin_ws/src/mes_detect_ball/include/weight_data/adaboost_ball_weight.txt", confusion(0, 0), confusion(1, 0));
}