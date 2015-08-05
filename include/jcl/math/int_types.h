//
//  int_types.h
//
//  Created by Jonathan Tompson on 4/23/12.
//
//  Integer types (formaly from math_types in jtil but broken out for indep.)
//

#pragma once

#include <stdint.h>  // For uint8_t, uint16_t, etc
#include <float.h>
#include <cmath>

#ifndef NULL
  #define NULL 0
#endif

#ifndef MAX_UINT32
  #define MAX_UINT32 0xffffffff
#endif
#ifndef MAX_UINT64
  #define MAX_UINT64 0xffffffffffffffff
#endif