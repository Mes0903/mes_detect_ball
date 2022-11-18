#pragma once
#ifndef __LABEL_H
#define __LABEL_H

/**
 * @file label.h
 * @author Mes (mes900903@gmail.com) (Discord: Mes#0903)
 * @brief labeling, THIS FILE DIDN'T BE USED!! I USED MATLAB TO LABEL NOW.
 * @version 0.1
 * @date 2022-11-17
 */

#include <Eigen/Eigen>

Eigen::VecotrXd ball_Label(const int section, const int S_n)
{
  Eigen::VectorXd PN = -1 * Eigen::VectorXd::Ones(S_n);

  if (section == 1) {
    PN(7) = 1;    // 第 7 個 segment 是球
    return PN;
  }
  elseif(section == 2)
  {
    PN(10) = 1;
    return PN;
  }
  elseif(section == 3)
  {
    PN(10) = 1;
    return PN;
  }
  elseif(section == 4)
  {
    PN(4) = 1;
    return PN;
  }
  elseif(section == 5)
  {
    PN(2) = 1;
    return PN;
  }
  elseif(section == 6)
  {
    PN(5) = 1;
    return PN;
  }
  elseif(section == 7)
  {
    PN(15) = 1;
    return PN;
  }
  elseif(section == 8)
  {
    PN(15) = 1;
    return PN;
  }
  elseif(section == 9)
  {
    PN(14) = 1;
    return PN;
  }
  elseif(section == 10)
  {
    PN(12) = 1;
    return PN;
  }
  elseif(section == 11)
  {
    PN(9) = 1;
    return PN;
  }
  elseif(section == 12)
  {
    PN(7) = 1;
    return PN;
  }
  elseif(section == 13)
  {
    PN(14) = 1;
    return PN;
  }
  elseif(section == 14)
  {
    PN(17) = 1;
    return PN;
  }
  elseif(section == 15)
  {
    PN(5) = 1;
    return PN;
  }
  elseif(section == 16)
  {
    PN(2) = 1;
    return PN;
  }
  elseif(section == 17)
  {
    PN(3) = 1;
    return PN;
  }
  elseif(section == 18)
  {
    PN(15) = 1;
    return PN;
  }
  elseif(section == 19)
  {
    PN(20) = 1;
    return PN;
  }
  elseif(section == 20)
  {
    PN(24) = 1;
    return PN;
  }
  elseif(section == 21)
  {
    PN(25) = 1;
    return PN;
  }
  elseif(section == 22)
  {
    PN(28) = 1;
    return PN;
  }
  elseif(section == 23)
  {
    PN(31) = 1;
    return PN;
  }
  elseif(section == 24)
  {
    PN(32) = 1;
    return PN;
  }
  elseif(section == 25)
  {
    PN(32) = 1;
    return PN;
  }
  elseif(section == 26)
  {
    PN(40) = 1;
    return PN;
  }
  elseif(section == 27)
  {
    PN(32) = 1;
    return PN;
  }
  elseif(section == 28)
  {
    PN(24) = 1;
    return PN;
  }
  elseif(section == 29)
  {
    PN(20) = 1;
    return PN;
  }
  elseif(section == 30)
  {
    PN(32) = 1;
    return PN;
  }
  elseif(section == 31)
  {
    PN(27) = 1;
    return PN;
  }
  elseif(section == 32)
  {
    PN(27) = 1;
    return PN;
  }
  elseif(section == 33)
  {
    PN(21) = 1;
    return PN;
  }
  elseif(section == 34)
  {
    PN(19) = 1;
    return PN;
  }
  elseif(section == 35)
  {
    PN(3) = 1;
    return PN;
  }
  elseif(section == 36)
  {
    PN(3) = 1;
    return PN;
  }
  elseif(section == 37)
  {
    PN(3) = 1;
    return PN;
  }
  elseif(section == 38)
  {
    PN(12) = 1;
    return PN;
  }
  elseif(section == 39)
  {
    PN(18) = 1;
    return PN;
  }
  elseif(section == 40)
  {
    PN(22) = 1;
    return PN;
  }
  elseif(section == 41)
  {
    PN(18) = 1;
    return PN;
  }
  elseif(section == 42)
  {
    PN(11) = 1;
    return PN;
  }
  elseif(section == 43)
  {
    PN(12) = 1;
    return PN;
  }
  elseif(section == 44)
  {
    PN(10) = 1;
    return PN;
  }
  elseif(section == 45)
  {
    PN(12) = 1;
    return PN;
  }
  elseif(section == 46)
  {
    PN(8) = 1;
    return PN;
  }
  elseif(section == 47)
  {
    PN(19) = 1;
    return PN;
  }
  elseif(section == 48)
  {
    PN(13) = 1;
    return PN;
  }
  elseif(section == 49)
  {
    PN(6) = 1;
    return PN;
  }
  elseif(section == 50)
  {
    PN(2) = 1;
    return PN;
  }
  elseif(section == 51)
  {
    PN(20) = 1;
    return PN;
  }
  elseif(section == 52)
  {
    PN(22) = 1;
    return PN;
  }
  elseif(section == 53)
  {
    PN(15) = 1;
    return PN;
  }
  elseif(section == 54)
  {
    PN(7) = 1;
    return PN;
  }
  elseif(section == 55)
  {
    PN(2) = 1;
    return PN;
  }
  elseif(section == 56)
  {
    PN(1) = 1;
    return PN;
  }
  elseif(section == 57)
  {
    PN(33) = 1;
    return PN;
  }
  elseif(section == 58)
  {
    PN(40) = 1;
    return PN;
  }
  elseif(section == 59)
  {
    PN(44) = 1;
    return PN;
  }
  elseif(section == 60)
  {
    PN(40) = 1;
    return PN;
  }
  elseif(section == 61)
  {
    PN(18) = 1;
    PN(20) = 1;
    return PN;
  }
  elseif(section == 62)
  {
    PN(15) = 1;
    PN(16) = 1;
    return PN;
  }
  elseif(section == 63)
  {
    PN(17) = 1;
    PN(19) = 1;
    return PN;
  }
  elseif(section == 64)
  {
    PN(15) = 1;
    return PN;
  }
  elseif(section == 65)
  {
    PN(17) = 1;
    PN(19) = 1;
    return PN;
  }
  elseif(section == 66)
  {
    PN(17) = 1;
    PN(19) = 1;
    return PN;
  }
  elseif(section == 67)
  {
    PN(17) = 1;
    PN(18) = 1;
    return PN;
  }
  elseif(section == 68)
  {
    PN(17) = 1;
    PN(18) = 1;
    return PN;
  }
  elseif(section == 69)
  {
    PN(18) = 1;
    PN(19) = 1;
    return PN;
  }
  elseif(section == 70)
  {
    PN(16) = 1;
    PN(17) = 1;
    return PN;
  }
  elseif(section == 71)
  {
    PN(18) = 1;
    PN(20) = 1;
    return PN;
  }
  elseif(section == 72)
  {
    PN(17) = 1;
    PN(19) = 1;
    return PN;
  }
  elseif(section == 73)
  {
    PN(17) = 1;
    return PN;
  }
  elseif(section == 74)
  {
    PN(17) = 1;
    PN(19) = 1;
    return PN;
  }
  elseif(section == 75)
  {
    PN(16) = 1;
    PN(18) = 1;
    return PN;
  }
  elseif(section == 76)
  {
    PN(18) = 1;
    PN(19) = 1;
    return PN;
  }
  elseif(section == 77)
  {
    PN(17) = 1;
    PN(19) = 1;
    return PN;
  }
  elseif(section == 78)
  {
    PN(16) = 1;
    PN(18) = 1;
    return PN;
  }
  elseif(section == 79)
  {
    PN(18) = 1;
    PN(20) = 1;
    return PN;
  }
  elseif(section == 80)
  {
    PN(16) = 1;
    PN(20) = 1;
    return PN;
  }
  elseif(section == 81)
  {
    PN(16) = 1;
    PN(18) = 1;
    return PN;
  }
  elseif(section == 82)
  {
    PN(13) = 1;
    PN(17) = 1;
    return PN;
  }
  elseif(section == 83)
  {
    PN(14) = 1;
    PN(16) = 1;
    return PN;
  }
  elseif(section == 84)
  {
    PN(13) = 1;
    PN(15) = 1;
    return PN;
  }
  elseif(section == 85)
  {
    PN(13) = 1;
    PN(16) = 1;
    return PN;
  }
  elseif(section == 86)
  {
    PN(14) = 1;
    PN(16) = 1;
    return PN;
  }
  elseif(section == 87)
  {
    PN(13) = 1;
    PN(15) = 1;
    return PN;
  }
  elseif(section == 88)
  {
    PN(9) = 1;
    PN(10) = 1;
    return PN;
  }
  elseif(section == 89)
  {
    PN(7) = 1;
    PN(9) = 1;
    return PN;
  }
  elseif(section == 90)
  {
    PN(7) = 1;
    PN(9) = 1;
    return PN;
  }
  elseif(section == 91)
  {
    PN(8) = 1;
    return PN;
  }
  elseif(section == 92)
  {
    PN(5) = 1;
    PN(7) = 1;
    return PN;
  }
  elseif(section == 93)
  {
    PN(5) = 1;
    PN(24) = 1;
    return PN;
  }
  elseif(section == 94)
  {
    PN(17) = 1;
    PN(18) = 1;
    return PN;
  }
  elseif(section == 95)
  {
    PN(17) = 1;
    PN(19) = 1;
    return PN;
  }
  elseif(section == 96)
  {
    PN(16) = 1;
    PN(18) = 1;
    return PN;
  }
  elseif(section == 97)
  {
    PN(13) = 1;
    return PN;
  }
  elseif(section == 98)
  {
    PN(11) = 1;
    PN(12) = 1;
    return PN;
  }
  elseif(section == 99)
  {
    PN(12) = 1;
    PN(15) = 1;
    return PN;
  }
  elseif(section == 100)
  {
    PN(13) = 1;
    PN(15) = 1;
    return PN;
  }
  elseif(section == 101)
  {
    PN(18) = 1;
    PN(20) = 1;
    return PN;
  }
  elseif(section == 102)
  {
    PN(17) = 1;
    PN(13) = 1;
    return PN;
  }
  elseif(section == 103)
  {
    PN(12) = 1;
    PN(14) = 1;
    return PN;
  }
  elseif(section == 104)
  {
    PN(9) = 1;
    PN(12) = 1;
    return PN;
  }
  elseif(section == 105)
  {
    PN(9) = 1;
    return PN;
  }
  elseif(section == 106)
  {
    PN(10) = 1;
    PN(12) = 1;
    return PN;
  }
  elseif(section == 107)
  {
    PN(10) = 1;
    PN(15) = 1;
    return PN;
  }
  elseif(section == 108)
  {
    PN(13) = 1;
    PN(18) = 1;
    return PN;
  }
  elseif(section == 109)
  {
    PN(10) = 1;
    PN(14) = 1;
    return PN;
  }
  elseif(section == 110)
  {
    PN(9) = 1;
    PN(10) = 1;
    return PN;
  }
  elseif(section == 111)
  {
    PN(13) = 1;
    PN(17) = 1;
    return PN;
  }
  elseif(section == 112)
  {
    PN(16) = 1;
    PN(19) = 1;
    return PN;
  }
  elseif(section == 113)
  {
    PN(13) = 1;
    PN(15) = 1;
    return PN;
  }
  elseif(section == 114)
  {
    PN(13) = 1;
    PN(15) = 1;
    return PN;
  }
  elseif(section == 115)
  {
    PN(16) = 1;
    PN(17) = 1;
    return PN;
  }
  elseif(section == 116)
  {
    PN(17) = 1;
    PN(19) = 1;
    return PN;
  }
  elseif(section == 117)
  {
    PN(18) = 1;
    PN(19) = 1;
    return PN;
  }
  elseif(section == 118)
  {
    PN(17) = 1;
    PN(19) = 1;
    return PN;
  }
  elseif(section == 119)
  {
    PN(16) = 1;
    PN(18) = 1;
    return PN;
  }
  elseif(section == 120)
  {
    PN(18) = 1;
    PN(20) = 1;
    return PN;
  }
}

#endif