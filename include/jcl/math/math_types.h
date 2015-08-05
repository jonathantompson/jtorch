//
//  math_types.h
//
//  Created by Jonathan Tompson on 4/23/12.
//
//  Vector types.
//
//  ****** Originally from my jtil library (but pulled out for jcl to reduce
//  compilation dependencies).  ******
//

#include "jcl/math/int_types.h"
#include "jcl/math/math_base.h"

#pragma once

#if defined(WIN32) || defined(_WIN32)
  #include <random>  // For std::tr1::mt19937, and others
#endif
#if defined(__APPLE__)
  #include <tr1/random> 
#endif

#ifndef EPSILON
  #define EPSILON (2 * FLT_EPSILON)  // 2 times machine precision for float
#endif

#ifndef DEPSILON
  #define DEPSILON (2 * DBL_EPSILON)  // 2 times machine precision for float
#endif

#ifndef LOOSE_EPSILON
  #define LOOSE_EPSILON 0.000001f
#endif

#ifndef LOOSE_DEPSILON
  #define LOOSE_DEPSILON 0.00000000001
#endif

#ifndef M_E
  #define M_E     2.71828182845904523536
#endif
#ifndef M_PI
  #define M_PI    3.14159265358979323846
#endif
#ifndef M_PI_2
  #define M_PI_2  1.57079632679489661923
#endif
#ifndef M_PI_4
  #define M_PI_4  0.785398163397448309616
#endif
#ifndef PI_OVER_180
  #define PI_OVER_180 0.017453292519943295769236907684886  // 2*PI / 360
#endif
#ifndef PI_OVER_360
  #define PI_OVER_360 0.0087266462599716478846184538424431
#endif

#ifndef MAX_INT16
  #define MAX_INT16 32767  // 2^15 - 1
#endif

#ifndef MAX_UINT32
  #define MAX_UINT32 0xffffffff
#endif
#ifndef MAX_UINT64
  #define MAX_UINT64 0xffffffffffffffff
#endif

#if defined(WIN32) || defined(_WIN32)
  #define RAND_ENGINE std::tr1::mt19937
  #define NORM_DIST std::tr1::normal_distribution  // Usgae: NORM_DIST<float> dist;
  #define UNIF_DIST std::tr1::uniform_real_distribution
#endif
#ifdef __APPLE__
  #define RAND_ENGINE std::mt19937
  #define NORM_DIST std::normal_distribution
  #define UNIF_DIST std::uniform_real_distribution
#endif

#if defined(ROW_MAJOR) && defined(COLUMN_MAJOR)
  #error define either ROW_MAJOR or COLUMN_MAJOR but not both
#endif

#if !defined(ROW_MAJOR) && !defined(COLUMN_MAJOR)
  #error define either ROW_MAJOR or COLUMN_MAJOR but not both
#endif
