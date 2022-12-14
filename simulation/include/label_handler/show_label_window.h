#ifndef SHOW_LABEL_WINDOW_H__
#define SHOW_LABEL_WINDOW_H__

#include "imgui_header.h"
#include "show_label_window.h"

#include <fstream>

void ShowLabelInformation(const std::string &filepath, std::fstream &feature_file, std::fstream &label_file);
void ShowLabel(std::ifstream &infile, const std::string &filepath, const int max_frame);

#endif