#ifndef LABEL_ANIMATE_INFO_H__
#define LABEL_ANIMATE_INFO_H__

#include "animate_info.h"

struct LabelAnimationInfo : public AnimationInfo {
  float label_mouse_area = static_cast<float>(0.05);
  bool save_label = false;
  bool enable_enter_save = false;
  bool output_txt = false;
  bool show_rect = false;
  bool auto_label = false;
  int current_save_frame = -1;
  int writed_max_frame = 0;
};

#endif