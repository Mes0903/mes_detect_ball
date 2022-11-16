#include "adaboost_classifier.h"
#include "load_matrix.h"
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

  if (case_num == 1) {
    std::cout << "input sample numbers\n>";
    std::cin >> sample;
  }
  std::cin.clear();
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  /* fitting */
  puts("read training data...");
  Eigen::MatrixXd train_X = readDataSet(filepath + "/include/dataset/box_train_xn.txt", 2888, 5);    // file, row, col

  puts("reading testing data...");
  Eigen::MatrixXd test_X = readDataSet(filepath + "/include/dataset/box_test_xn.txt", 1238, 5);

  puts("reading training label...");
  Eigen::VectorXd train_Y = readLabel(filepath + "/include/dataset/box_train_yn.txt", 2888);    // file, segment num(row)

  puts("reading testing label...");
  Eigen::VectorXd test_Y = readLabel(filepath + "/include/dataset/box_test_yn.txt", 1238);


  if (case_num == 1) {
    Normalizer normalizer;
    normalizer.fit(train_X);
    train_X = normalizer.transform(train_X);
    test_X = normalizer.transform(test_X);

    for (int i = 0; i < sample; ++i) {
      puts("start tranning");
      Adaboost A(100);
      A.fit(train_X, train_Y);

      // prediction
      puts("make prediction");
      Eigen::VectorXd pred_Y = A.predict(test_X);

      puts("cal confusion matrix");
      Eigen::MatrixXd confusion = A.cal_confusion_matrix(test_Y, pred_Y);
      std::cout << confusion << '\n';

      Weight_handle::store_weight(filepath + "/include/weight_data/adaboost_box_weight.txt", A, normalizer);
    }
  }
  else {
    Normalizer normalizer;
    Adaboost A;
    Weight_handle::load_weight(filepath + "/include/weight_data/adaboost_ball_weight.txt", A, normalizer);

    train_X = normalizer.transform(train_X);
    test_X = normalizer.transform(test_X);

    puts("make prediction");
    Eigen::VectorXd pred_Y = A.predict(test_X);
    puts("cal confusion matrix");
    Eigen::MatrixXd confusion = A.cal_confusion_matrix(test_Y, pred_Y);
    std::cout << confusion << '\n';
  }
}