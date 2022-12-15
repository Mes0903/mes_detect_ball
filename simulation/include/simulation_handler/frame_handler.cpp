#include "frame_handler.h"

#include <Eigen/Eigen>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

extern int frame, max_frame;
extern Eigen::MatrixXd xy_data;

void transform_frame(const std::string &in_filepath, const std::string &out_filepath)
{
  max_frame = 0;

  std::ifstream infile(in_filepath);
  if (infile.fail()) {
    std::cerr << "cant found " << in_filepath << '\n';
    exit(1);
  }

  std::ofstream outfile(out_filepath, std::ios::binary);
  if (outfile.fail()) {
    std::cerr << "cant found " << out_filepath << '\n';
    exit(1);
  }

  std::string line;
  std::stringstream ss;
  double x, y;
  while (std::getline(infile, line)) {
    ++max_frame;

    ss << line;
    ss >> x >> y;

    outfile.write(reinterpret_cast<char *>(&x), sizeof(double));
    outfile.write(reinterpret_cast<char *>(&y), sizeof(double));

    ss.str("");
    ss.clear();
  }

  max_frame /= 720;
  --max_frame;
}

void read_frame(std::ifstream &infile)
{
  infile.seekg(frame * sizeof(double) * 2 * 720, std::ios::beg);

  for (int i{}; i < 720; ++i) {
    infile.read(reinterpret_cast<char *>(&xy_data(i, 0)), sizeof(double));
    infile.read(reinterpret_cast<char *>(&xy_data(i, 1)), sizeof(double));
  }
}