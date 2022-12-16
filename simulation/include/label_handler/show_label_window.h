#ifndef SHOW_LABEL_WINDOW_H__
#define SHOW_LABEL_WINDOW_H__
#include "imgui_header.h"
#include "show_label_window.h"
#include "frame_handler.h"

#include <fstream>

void ShowLabelInformation();
void ShowLabel(std::ifstream &infile);

#endif