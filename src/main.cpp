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
  int case_t_l = 0, case_save = 0;
  std::cout << "Input the number to choose case: \n1: trainning data\n2: load data\n> ";
  std::cin >> case_t_l;

  uint32_t Iteration = 0;
  if (case_t_l == 1)
  {
    std::cout << "Input the sample number you want: ";
    std::cin >> Iteration;

    std::cout << "Input 1 if you want to save the training data\n> ";
    std::cin >> case_save;
  }

  std::cin.clear();
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  /* fitting */
  puts("read training data...");
  Eigen::MatrixXd train_data = readDataSet("/home/mes/catkin_ws/src/mes_detect_ball/include/dataset/xy_data2.txt", 2248, 2); // file, row, col
  // puts("transforming training data to feature...");
  // train_data = transform_to_feature(train_data);

  // puts("reading X2");
  // Eigen::MatrixXd test_data = readDataSet("/home/mes/catkin_ws/src/mes_detect_ball/include/dataset/test_X.txt", 1529, 5);

  puts("reading training label...");
  Eigen::VectorXd train_label = readLabel("/home/mes/catkin_ws/src/mes_detect_ball/include/dataset/xy_label.txt", 2248); // file, segment num(row)

  // puts("reading Y2");
  // Eigen::VectorXd test_Y = readLabel("/home/mes/catkin_ws/src/mes_detect_ball/include/dataset/test_Y.txt", 1529);

  if (case_t_l == 1)
  {
    for (uint32_t i = 0; i < Iteration; ++i)
    {
      puts("start tranning");
      Adaboost A(100);
      A.fit(train_data, train_label);

      // prediction
      puts("make prediction");
      Eigen::VectorXd pred_Y = A.predict(train_data);

      puts("cal confusion matrix");
      Eigen::MatrixXd confusion = cal_confusion_matrix(train_label, pred_Y);

      std::cout << confusion << '\n';
      if (case_save == 1)
        A.store_weight("/home/mes/catkin_ws/src/mes_detect_ball/include/weight_data/adaboost_weight.txt", confusion(0, 0), confusion(1, 0));
    }
  }
  else if (case_t_l == 2)
  {
    Adaboost A;
    puts("start loading weight data");
    A.load_weight("/home/mes/catkin_ws/src/mes_detect_ball/include/weight_data/adaboost_weight.txt");

    // prediction
    puts("make prediction");
    Eigen::VectorXd pred_Y = A.predict(train_data);

    puts("cal confusion matrix");
    Eigen::MatrixXd confusion = cal_confusion_matrix(train_label, pred_Y);

    std::cout << confusion << '\n';
  }
  else
  {
    std::cerr << "wrong case number, process will exit\n";
    exit(1);
  }
}