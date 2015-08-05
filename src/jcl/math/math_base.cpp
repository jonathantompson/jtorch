#include <stdlib.h>
#include <cmath>
#include <cstddef>
#include "jcl/math/math_base.h"

#define rand_r rand

namespace jcl {
namespace math {

  // Brute force primality test, from: 
  // http://stackoverflow.com/questions/4475996/given-prime-number-n-compute-the-next-prime
  // Answer 7 by Howard Hinnant --> Implementation 3
  bool IsPrime(std::size_t x) {
    if (x != 2 && x % 2 == 0) {
      return false;
    }
    for (std::size_t i = 3; true; i += 2) {
      std::size_t q = x / i;
      if (q < i) {
        return true;
      }
      if (x == q * i) {
        return false;
      }
    }
    return true;
  }

  // Brute force next prime integer (reasonably quick I think)
  // http://stackoverflow.com/questions/4475996/given-prime-number-n-compute-the-next-prime
  // Answer 7 by Howard Hinnant --> Implementation 4
  std::size_t NextPrime(std::size_t x) {
    if (x <= 2)
      return 2;
    if (!(x & 1))
      ++x;
    for (; !IsPrime(x); x += 2)
      ;
    return x;
  }

  void Convolve(const float* input, const float* kernel, float* output,
    const int32_t in_width, const int32_t in_height,
    const int32_t out_width, const int32_t out_height, 
    const int32_t kernel_size, const int32_t n_threads) {
    // Check the sizes (in case the user messed up)
    if (out_width != in_width - kernel_size + 1 ||
      out_height != in_height - kernel_size + 1) {
      throw std::runtime_error("jcl::math::Convolve() - ERROR: "
        "Input/Output size mismatch!");
   } 
    // From:
    // http://developer.amd.com/resources/heterogeneous-computing/opencl-zone/programming-in-opencl/image-convolution-using-opencl/
    #pragma omp parallel for num_threads(n_threads)
    for (int32_t yOut = 0; yOut < out_height; yOut++) {
      const int32_t yInTopLeft = yOut;
      for (int32_t xOut = 0; xOut < out_width; xOut++) {
        const int32_t xInTopLeft = xOut;
        float sum = 0;
        for (int32_t r = 0; r < kernel_size; r++) {
          const int32_t idxFtmp = r * kernel_size;
          const int32_t yIn = yInTopLeft + r;
          const int32_t idxIntmp = yIn * in_width + xInTopLeft;
          for (int32_t c = 0; c < kernel_size; c++) {
            const int32_t idxF  = idxFtmp  + c;
            const int32_t idxIn = idxIntmp + c;    
            sum += kernel[idxF] * input[idxIn];
          }
        } //for (int r = 0...

        const int idxOut = yOut * out_width + xOut;
        output[idxOut] = sum;

      } //for (int xOut = 0...
    } //for (int yOut = 0...
  }

}  // namespace math
}  // namespace jcl
