#ifndef SHOW_SIMULATION_WINDOW_H__
#define SHOW_SIMULATION_WINDOW_H__

#include <fstream>

void ShowSimulationInformation();
void ShowSimulation(std::ifstream &infile, const int max_frame);

#endif