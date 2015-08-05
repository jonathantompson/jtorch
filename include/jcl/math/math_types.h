//
//  math_types.h
//
//  Created by Jonathan Tompson on 4/23/12.
//
//  Vector types.
//

#include "jcl/math/int_types.h"
#include "jcl/math/math_base.h"

#pragma once

#if defined(WIN32) || defined(_WIN32)
  #include <random>  // For std::tr1::mt19937, and others
#else
  #include <tr1/random> 
#endif

namespace {
  static const float kEpsilon = 2 * FLT_EPSILON;
  static const double kDEpsilon = 2 * DBL_EPSILON;
  static const float kLooseEpsilon = 0.000001f;
  static const double kLooseDEpsilon = 0.00000000001;
  static const double kME = 2.71828182845904523536;
  static const double kMPi = 3.14159265358979323846;
  static const double kMPi2 = 1.57079632679489661923;
  static const double kMPi4 = 0.785398163397448309616;
  static const double kPiOver180 = 0.017453292519943295769236907684886;
  static const double kPiOver360 = 0.0087266462599716478846184538424431;
};  // unnamed namespace

typedef std::tr1::mt19937 RandEngine;
// The following pre-processor defines are ugly, but unfortunately
// Visual Studio does not support aliased declarations.
#if defined(WIN32) || defined(_WIN32)
  #define NORM_DIST std::tr1::normal_distribution  // Usage: NORM_DIST<float> dist;
  #define UNIF_DIST std::tr1::uniform_real_distribution
#else
  #define NORM_DIST std::normal_distribution
  #define UNIF_DIST std::uniform_real_distribution
#endif