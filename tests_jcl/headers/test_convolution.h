//
//  test_math.cpp
//
//  Created by Jonathan Tompson on 3/14/12.
//
//  C++ code to implement test_math.m --> Use matlab to check C results
//  Typical code will not use templates directly, but I do so here so
//  that I can switch out float for doubles and test both cases
//
//  LOTS of tests here.  There's a lot of code borrowed from Prof 
//  Alberto Lerner's code repository (I took his distributed computing class, 
//  which was amazing), particularly the test unit stuff.

#include "jcl/math/math_types.h"
#include "jcl/math/math_base.h"
#include "jcl/jcl.h"
#include "clk/clk.h"

// convolution_kernel.cl
const char* conv_kernel_c_str =
"__kernel void Convolve(const __global float * pInput, "
"  __constant float * pFilter, __global  float * pOutput, const int nInWidth, "
"  const int nFilterWidth) {"
"  const int nWidth = get_global_size(0);"
"  const int xOut = get_global_id(0);"
"  const int yOut = get_global_id(1);"
"  const int xInTopLeft = xOut;"
"  const int yInTopLeft = yOut;"
"  float sum = 0;"
"  for (int r = 0; r < nFilterWidth; r++) {"
"    const int idxFtmp = r * nFilterWidth;"
"    const int yIn = yInTopLeft + r;"
"    const int idxIntmp = yIn * nInWidth + xInTopLeft;"
"    for (int c = 0; c < nFilterWidth; c++) {"
"      const int idxF  = idxFtmp  + c;"
"      const int idxIn = idxIntmp + c;"
"      sum += pFilter[idxF]*pInput[idxIn];"
"    }"
"  }"
"  const int idxOut = yOut * nWidth + xOut;"
"  pOutput[idxOut] = sum;"
"}";

using namespace jcl::math;
using namespace clk;
using namespace jcl;

TEST(OpenCLTests, TestConvolution) {
  try {
    // Allocate a Random input image and a random kernel
    const int32_t src_width = 139;
    const int32_t src_height = 139;
    const int32_t kernel_size = 12;
    const int32_t n_threads = 8;
    const int32_t dst_width = src_width - kernel_size + 1;  // 64
    const int32_t dst_height = src_height - kernel_size + 1;  // 64
    const uint32_t num_repeats = 100;

    float* input = new float[src_width * src_height];
    float* kernel = new float[kernel_size * kernel_size];
    float* output = new float[dst_width * dst_height];
    float* outputcl = new float[dst_width * dst_height];

    RAND_ENGINE eng;
    NORM_DIST<float> norm_dist;
    eng.seed();
    Clk clk;

    for (uint32_t uv = 0; uv < src_width * src_height; uv++) {
      input[uv] = norm_dist(eng);
    }

    double dFilterSum = 0;
    for (uint32_t uv = 0; uv < kernel_size * kernel_size; uv++) {
      kernel[uv] = fabsf(norm_dist(eng));
      dFilterSum += (double)kernel[uv];
    }
    for (uint32_t uv = 0; uv < kernel_size * kernel_size; uv++) {
      kernel[uv] /= (float)dFilterSum;  // Normalize the random filter
    }

    // Call the CPU version
    Convolve(input, kernel, output, src_width, src_height, dst_width, 
      dst_height, kernel_size, n_threads);
    std::cout << std::endl;

    // Create an OpenCL context on each of the devices
    CLDevice dev[2] = {CLDeviceCPU, CLDeviceGPU};
    for (uint32_t d = 0; d < 2; d++) {
      if (!JCL::queryDeviceExists(dev[d], CLVendorAny)) {
        std::cout << "\tOpenCL Device '" << d << "' does not exist. ";
        std::cout << "\tSkipping test" << std::endl;
        continue;
      }
      JCL* context = new JCL(dev[d], CLVendorAny);
      const uint32_t dev_id = 0;
      std::cout << "\tUsing OpenCL Device: " << context->getDeviceName(dev_id);
      std::cout << std::endl;

      // Initialize Buffers
      JCLBuffer input_buffer = context->allocateBuffer(CLBufferTypeRead, 
        src_width * src_height);
      JCLBuffer kernel_buffer = context->allocateBuffer(CLBufferTypeRead, 
        kernel_size * kernel_size);
      JCLBuffer output_buffer = context->allocateBuffer(CLBufferTypeWrite, 
        dst_width * dst_height);
      context->writeToBuffer(input, dev_id, input_buffer, true);
      context->writeToBuffer(kernel, dev_id, kernel_buffer, true);

      context->useKernelCStr(conv_kernel_c_str, "Convolve");
      context->setArg(0, input_buffer);
      context->setArg(1, kernel_buffer);
      context->setArg(2, output_buffer);
      context->setArg(3, src_width);
      context->setArg(4, kernel_size);

      uint32_t dim = 2;
      uint32_t global_worksize[2] = {dst_width, dst_height};
      
      // Force a particular local worksize:
      uint32_t local_worksize[2] = {16, 16};
      uint32_t max_itemsize[3];
      for (uint32_t i = 0; i < 3; i++) {
        max_itemsize[i] = context->getMaxWorkitemSize(dev_id, i);
      }
      for (uint32_t i = 0; i < 2; i++) {
        local_worksize[i] = std::min<int32_t>(local_worksize[i],                                     
          max_itemsize[i]);
      }
      context->runKernel(dev_id, dim, global_worksize, local_worksize, true);
      
      // Let OpenCL choose the local worksize
      // context->runKernel2D(dev_id, global_worksize, true);
      
      context->readFromBuffer(outputcl, dev_id, output_buffer, true);

      // Clean up the context
      delete context;

      // Check that the results are the same:
      for (uint32_t i = 0; i < dst_width * dst_height; i++) {
        EXPECT_APPROX_EQ(outputcl[i], output[i]);
        if (fabsf(outputcl[i] - output[i]) > 0.000001) {
          std::cout << "\tData mismatch on device " << d << "!" << std::endl;
          // We only need to show one of these are wrong...
          break;
        }
      }
    }
  
    // Clean up
    delete[] input;
    delete[] kernel;
    delete[] output;
    delete[] outputcl;
  } catch (std::runtime_error err) {
    std::cout << "\tException thrown: " << err.what() << std::endl;
    EXPECT_TRUE(false);
  }
}

TEST(OpenCLTests, ProfileConvolution) {
  try {
    // Allocate a Random input image and a random kernel
    const int32_t src_width = 139;
    const int32_t src_height = 139;
    const int32_t kernel_size = 12;
    const int32_t n_threads = 8;
    const int32_t dst_width = src_width - kernel_size + 1;  // 64
    const int32_t dst_height = src_height - kernel_size + 1;  // 64
    const uint32_t num_repeats = 1000;

    float* input = new float[src_width * src_height];
    float* kernel = new float[kernel_size * kernel_size];
    float* output = new float[dst_width * dst_height];

    RAND_ENGINE eng;
    NORM_DIST<float> norm_dist;
    eng.seed();
    Clk clk;

    for (uint32_t uv = 0; uv < src_width * src_height; uv++) {
      input[uv] = norm_dist(eng);
    }

    double dFilterSum = 0;
    for (uint32_t uv = 0; uv < kernel_size * kernel_size; uv++) {
      kernel[uv] = fabsf(norm_dist(eng));
      dFilterSum += (double)kernel[uv];
    }
    for (uint32_t uv = 0; uv < kernel_size * kernel_size; uv++) {
      kernel[uv] /= (float)dFilterSum;  // Normalize the random filter
    }

    // Call the CPU version
    double cpu_t_accum = 0;
    for (uint32_t i = 0; i < num_repeats; i++) {
      double t0 = clk.getTime();
      Convolve(input, kernel, output, src_width, src_height, dst_width, 
        dst_height, kernel_size, n_threads);
      double t1 = clk.getTime();
      cpu_t_accum += (t1 - t0);

      // Add the input to the output so that the optimizer doesn't delete the 
      // loop entirely.
      for (uint32_t uv = 0; uv < dst_width * dst_height; uv++) {
        input[uv] += output[uv];
      }
    }
    std::cout << std::endl << "\tCPU time = " << cpu_t_accum << std::endl;

    // Create an OpenCL context on the GPU if possible, otherwise the CPU
    JCL* context = NULL; 
    bool gpu = false;
    if (JCL::queryDeviceExists(CLDeviceGPU, CLVendorAny)) {
      context = new JCL(CLDeviceGPU, CLVendorAny);
      gpu = true;
    } else if (JCL::queryDeviceExists(CLDeviceCPU, CLVendorAny)) {
      context = new JCL(CLDeviceCPU, CLVendorAny);
    }
    if ( context != NULL ) {
      const uint32_t dev_id = 0;
      std::cout << "\tUsing OpenCL Device: " << context->getDeviceName(dev_id);
      std::cout << std::endl;
  
      float* outputcl = new float[dst_width * dst_height];
      JCLBuffer input_buffer = context->allocateBuffer(CLBufferTypeRead, 
        src_width * src_height);
      JCLBuffer kernel_buffer = context->allocateBuffer(CLBufferTypeRead, 
        kernel_size * kernel_size);
      JCLBuffer output_buffer = context->allocateBuffer(CLBufferTypeWrite, 
        dst_width * dst_height);
      context->writeToBuffer(input, dev_id, input_buffer, true);
      context->writeToBuffer(kernel, dev_id, kernel_buffer, true);
  
      context->useKernelCStr(conv_kernel_c_str, "Convolve");
      context->setArg(0, input_buffer);
      context->setArg(1, kernel_buffer);
      context->setArg(2, output_buffer);
      context->setArg(3, src_width);
      context->setArg(4, kernel_size);
  
      // Call it once to compile the kernel

      uint32_t dim = 2;
      uint32_t global_worksize[2] = {dst_width, dst_height};
      
      // Force a particular local worksize:
      uint32_t local_worksize[2] = {16, 16};
      uint32_t max_itemsize[3];
      for (uint32_t i = 0; i < 3; i++) {
        max_itemsize[i] = context->getMaxWorkitemSize(dev_id, i);
      }
      for (uint32_t i = 0; i < 2; i++) {
        local_worksize[i] = std::min<int32_t>(local_worksize[i],                                     
          max_itemsize[i]);
      }
      context->runKernel(dev_id, dim, global_worksize, local_worksize, true);
      context->sync(dev_id);
  
      // Try one where we let OpenCL choose the workgroup size
      context->runKernel(dev_id, dim, global_worksize, true);
      context->sync(dev_id);
  
      double t0 = clk.getTime();
      for (uint32_t i = 0; i < num_repeats; i++) {
        context->runKernel(dev_id, dim, global_worksize, local_worksize, false);
      }
      context->readFromBuffer(outputcl, dev_id, output_buffer, false);
      context->sync(dev_id);
      double t1 = clk.getTime();
      std::cout << "\tGPU time (manual sizes) = " << (t1 - t0) << std::endl;
  
      t0 = clk.getTime();
      for (uint32_t i = 0; i < num_repeats; i++) {
        context->runKernel(dev_id, dim, global_worksize, false);
      }
      context->readFromBuffer(outputcl, dev_id, output_buffer, false);
      context->sync(dev_id);
      t1 = clk.getTime();
      std::cout << "\tGPU time (opencl sizes) = " << (t1 - t0) << std::endl;
      
      delete context;
      delete outputcl;
    }

    // Clean up
    delete input;
    delete kernel;
    delete output;

  } catch (std::runtime_error err) {
    std::cout << "\tException thrown: " << err.what() << std::endl;
    EXPECT_TRUE(false);
  }
}
