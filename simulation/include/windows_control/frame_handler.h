#ifndef FRAME_HANDLER_H__
#define FRAME_HANDLER_H__

#include <Eigen/Eigen>
#include <string>
#include <fstream>

void transform_frame(const std::string &in_filepath, const std::string &out_filepath);
void read_frame(std::ifstream &infile);

#endif