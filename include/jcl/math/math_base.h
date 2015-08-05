//
//  math_base.h
//
//  Created by Jonathan Tompson on 3/14/12.
//
//  This is where all the miscilaneous math functions go.  It's a little messy.
//
//  ****** Originally from my jtil library (but pulled out for jcl to reduce
//  compilation dependencies).  ******
//

#include "jcl/math/int_types.h"

#pragma once

// Define either ROW_MAJOR (DirectX) or COLUMN_MAJOR (openGL)
// #define ROW_MAJOR
#define COLUMN_MAJOR

#ifdef __APPLE__
  #include <algorithm>
#endif
#include <stdexcept>
#include <string>

namespace jcl {
namespace math {
  
  std::size_t NextPrime(std::size_t x);  // Return the next prime number:
  bool IsPrime(std::size_t x);  // Brute-force primality test

  // Convolve - Basic 2D convolution
  // Kernel is NxN (square)
  // Output dimensions are: 
  //   out_width = in_width - kernel_size + 1
  //   out_height = in_height - kernel_size + 1
  void Convolve(const float* input, const float* kernel, float* output,
    const int32_t in_width, const int32_t in_height,
    const int32_t out_width, const int32_t out_height, 
    const int32_t kernel_size, const int32_t n_threads);

};  // namespace math
};  // namespace jcl
