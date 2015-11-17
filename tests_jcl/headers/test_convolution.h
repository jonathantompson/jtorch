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
#include "jcl/opencl_context.h"
#include "clk/clk.h"

// convolution_kernel.cl
static const char* conv_kernel_c_str =
    "__kernel void Convolve(const __global float * pInput, "
    "  __constant float * pFilter, __global  float * pOutput, const int "
    "nInWidth, "
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

TEST(OpenCLTests, TestConvolution) {
  // Allocate a Random input image and a random kernel
  const int32_t src_width = 139;
  const int32_t src_height = 139;
  const int32_t kernel_wh = 12;
  const int32_t n_threads = 8;
  const int32_t dst_width = src_width - kernel_wh + 1;    // 64
  const int32_t dst_height = src_height - kernel_wh + 1;  // 64
  const uint32_t num_repeats = 100;

  const int32_t input_sz = src_width * src_height;
  std::unique_ptr<float[]> input(new float[input_sz]);
  const int32_t kernel_sz = kernel_wh * kernel_wh;
  std::unique_ptr<float[]> kernel(new float[kernel_sz]);
  const int32_t output_sz = dst_width * dst_height;
  std::unique_ptr<float[]> output(new float[output_sz]);
  std::unique_ptr<float[]> outputcl(new float[output_sz]);

  RandEngine eng;
  NORM_DIST<float> norm_dist;
  eng.seed();
  clk::Clk clk;

  for (uint32_t uv = 0; uv < input_sz; uv++) {
    input[uv] = norm_dist(eng);
  }

  double dFilterSum = 0;
  for (uint32_t uv = 0; uv < kernel_sz; uv++) {
    kernel[uv] = fabsf(norm_dist(eng));
    dFilterSum += (double)kernel[uv];
  }
  for (uint32_t uv = 0; uv < kernel_sz; uv++) {
    kernel[uv] /= (float)dFilterSum;  // Normalize the random filter
  }

  // Call the CPU version
  jcl::math::Convolve(input.get(), kernel.get(), output.get(), src_width,
                      src_height, dst_width, dst_height, kernel_wh, n_threads);

  // Create an OpenCL context on all devices
  EXPECT_TRUE(jcl::OpenCLContext::queryDeviceExists(jcl::CLDeviceAll,
                                                    jcl::CLVendorAny));
  std::unique_ptr<jcl::OpenCLContext> context(new jcl::OpenCLContext());
  const bool verbose_startup = false;
  context->init(jcl::CLDeviceAll, jcl::CLVendorAny, verbose_startup);

  const uint32_t num_devices = context->getNumDevices();
  EXPECT_GT(num_devices, 0);

  // Test on all devices
  for (uint32_t dev_id = 0; dev_id < num_devices; dev_id++) {
    // Initialize Buffers
    std::shared_ptr<jcl::OpenCLBufferData> input_buffer =
        context->allocateBuffer(jcl::CLBufferTypeRead, input_sz);
    std::shared_ptr<jcl::OpenCLBufferData> kernel_buffer =
        context->allocateBuffer(jcl::CLBufferTypeRead, kernel_sz);
    std::shared_ptr<jcl::OpenCLBufferData> output_buffer =
        context->allocateBuffer(jcl::CLBufferTypeWrite, output_sz);
    context->writeToBuffer(input.get(), input_sz, dev_id, input_buffer, true);
    context->writeToBuffer(kernel.get(), kernel_sz, dev_id, kernel_buffer,
                           true);

    context->useKernelCStr(conv_kernel_c_str, "Convolve");
    context->setArg(0, input_buffer);
    context->setArg(1, kernel_buffer);
    context->setArg(2, output_buffer);
    context->setArg(3, src_width);
    context->setArg(4, kernel_wh);

    uint32_t dim = 2;
    uint32_t global_worksize[2] = {dst_width, dst_height};

    // Force a particular local worksize:
    uint32_t local_worksize[2] = {16, 16};
    uint32_t max_itemsize[3];
    for (uint32_t i = 0; i < 3; i++) {
      max_itemsize[i] = context->getMaxWorkitemSize(dev_id, i);
    }
    for (uint32_t i = 0; i < 2; i++) {
      local_worksize[i] = std::min<int32_t>(local_worksize[i], max_itemsize[i]);
    }
    context->runKernel(dev_id, dim, global_worksize, local_worksize, true);

    // Let OpenCL choose the local worksize
    // context->runKernel2D(dev_id, global_worksize, true);

    context->readFromBuffer(outputcl.get(), output_sz, dev_id, output_buffer,
                            true);

    // Check that the results are the same:
    for (uint32_t i = 0; i < output_sz; i++) {
      EXPECT_APPROX_EQ(outputcl[i], output[i], 1e-6);
      if (fabsf(outputcl[i] - output[i]) > 1e-6) {
        std::cout << "\tData mismatch on device " << dev_id << "!" << std::endl;
        // We only need to show one of these are wrong...
        break;
      }
    }
  }
}

TEST(OpenCLTests, ProfileConvolution) {
  // Allocate a Random input image and a random kernel
  const int32_t src_width = 139;
  const int32_t src_height = 139;
  const int32_t kernel_wh = 12;
  const int32_t n_threads = 8;
  const int32_t dst_width = src_width - kernel_wh + 1;    // 64
  const int32_t dst_height = src_height - kernel_wh + 1;  // 64
  const uint32_t num_repeats = 1000;

  const int32_t input_sz = src_width * src_height;
  std::unique_ptr<float[]> input(new float[input_sz]);
  const int32_t kernel_sz = kernel_wh * kernel_wh;
  std::unique_ptr<float[]> kernel(new float[kernel_sz]);
  const int32_t output_sz = dst_width * dst_height;
  std::unique_ptr<float[]> output(new float[output_sz]);

  RandEngine eng;
  NORM_DIST<float> norm_dist;
  eng.seed();
  clk::Clk clk;

  for (uint32_t uv = 0; uv < input_sz; uv++) {
    input[uv] = norm_dist(eng);
  }

  double dFilterSum = 0;
  for (uint32_t uv = 0; uv < kernel_sz; uv++) {
    kernel[uv] = fabsf(norm_dist(eng));
    dFilterSum += (double)kernel[uv];
  }
  for (uint32_t uv = 0; uv < kernel_sz; uv++) {
    kernel[uv] /= (float)dFilterSum;  // Normalize the random filter
  }

  // Call the CPU version
  double cpu_t_accum = 0;
  for (uint32_t i = 0; i < num_repeats; i++) {
    double t0 = clk.getTime();
    jcl::math::Convolve(input.get(), kernel.get(), output.get(), src_width,
                        src_height, dst_width, dst_height, kernel_wh,
                        n_threads);
    double t1 = clk.getTime();
    cpu_t_accum += (t1 - t0);

    // Add the input to the output so that the optimizer doesn't delete the
    // loop entirely.
    for (uint32_t uv = 0; uv < output_sz; uv++) {
      input[uv] += output[uv];
    }
  }
  std::cout << std::endl << "\tCPU time = " << cpu_t_accum << std::endl;

  // Create an OpenCL context on all devices
  EXPECT_TRUE(jcl::OpenCLContext::queryDeviceExists(jcl::CLDeviceAll,
                                                    jcl::CLVendorAny));
  std::unique_ptr<jcl::OpenCLContext> context(new jcl::OpenCLContext());
  const bool verbose_startup = false;
  context->init(jcl::CLDeviceAll, jcl::CLVendorAny, verbose_startup);

  const uint32_t num_devices = context->getNumDevices();
  EXPECT_GT(num_devices, 0);

  // Profile on all devices.
  for (uint32_t dev_id = 0; dev_id < num_devices; dev_id++) {
    std::unique_ptr<float[]> outputcl(new float[output_sz]);
    std::shared_ptr<jcl::OpenCLBufferData> input_buffer =
        context->allocateBuffer(jcl::CLBufferTypeRead, input_sz);
    std::shared_ptr<jcl::OpenCLBufferData> kernel_buffer =
        context->allocateBuffer(jcl::CLBufferTypeRead, kernel_sz);
    std::shared_ptr<jcl::OpenCLBufferData> output_buffer =
        context->allocateBuffer(jcl::CLBufferTypeWrite, output_sz);
    context->writeToBuffer(input.get(), input_sz, dev_id, input_buffer, true);
    context->writeToBuffer(kernel.get(), kernel_sz, dev_id, kernel_buffer,
                           true);

    context->useKernelCStr(conv_kernel_c_str, "Convolve");
    context->setArg(0, input_buffer);
    context->setArg(1, kernel_buffer);
    context->setArg(2, output_buffer);
    context->setArg(3, src_width);
    context->setArg(4, kernel_wh);

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
      local_worksize[i] = std::min<int32_t>(local_worksize[i], max_itemsize[i]);
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
    context->readFromBuffer(outputcl.get(), output_sz, dev_id, output_buffer,
                            false);
    context->sync(dev_id);
    double t1 = clk.getTime();
    std::cout << "\t device " << dev_id << " ("
              << context->getDeviceName(dev_id)
              << ") time (manual sizes) = " << (t1 - t0) << std::endl;

    t0 = clk.getTime();
    for (uint32_t i = 0; i < num_repeats; i++) {
      context->runKernel(dev_id, dim, global_worksize, false);
    }
    context->readFromBuffer(outputcl.get(), output_sz, dev_id, output_buffer,
                            false);
    context->sync(dev_id);
    t1 = clk.getTime();
    std::cout << "\t device " << dev_id << " ("
              << context->getDeviceName(dev_id)
              << ") time (opencl sizes) = " << (t1 - t0) << std::endl;
  }
}
