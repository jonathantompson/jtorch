//
//  test_memory.h
//
//  Created by Jonathan Tompson on 11/04/15.
//
//  Tests for jcl memory operations.

#define _USE_MATH_DEFINES
#include <math.h>

#include "jcl/math/math_types.h"
#include "jcl/math/math_base.h"
#include "jcl/opencl_context.h"

namespace {

static const char* kFillKernel =
    "    __kernel void Fill(\n"
    "      __global float* output,  /* 0 */\n"
    "      const float value) {     /* 1 */\n"
    "      const int x_out = get_global_id(0);\n"
    "      output[x_out] = value;\n"
    "    }";

void FillBuffer(jcl::OpenCLContext* cl_context, const uint32_t dev_id,
                const float fill_value,
                std::shared_ptr<jcl::OpenCLBufferData> buffer) {
  cl_context->useKernelCStr(kFillKernel, "Fill");
  cl_context->setArg(0, buffer);
  cl_context->setArg(1, fill_value);
  uint32_t dim = 1;
  uint32_t nelem = buffer->nelems();
  cl_context->runKernel(dev_id, dim, &nelem, false);
}

}  // unnamed namespace

TEST(OpenCLTests, TestSubBuffer) {
  // Allocate a Random input image and a random kernel
  const int32_t nchans = 3;
  const int32_t width = 11;
  const int32_t height = 13;
  const int32_t input_sz = nchans * height * width;

  // Create an OpenCL context on all devices
  EXPECT_TRUE(jcl::OpenCLContext::queryDeviceExists(jcl::CLDeviceAll,
                                                    jcl::CLVendorAny));
  std::unique_ptr<jcl::OpenCLContext> context(new jcl::OpenCLContext());
  const bool verbose_startup = false;
  context->init(jcl::CLDeviceAll, jcl::CLVendorAny, verbose_startup);

  const uint32_t num_devices = context->getNumDevices();
  EXPECT_GT(num_devices, 0);

  // Test on all devices.
  for (uint32_t dev_id = 0; dev_id < num_devices; dev_id++) {
    // Initialize a buffer.
    std::shared_ptr<jcl::OpenCLBufferData> buffer =
        context->allocateBuffer(jcl::CLBufferTypeRead, input_sz);

    // Fill the buffer with ones
    FillBuffer(context.get(), dev_id, 1, buffer);

    // Get a sub-buffer to the blue channel.
    const uint32_t nelems = width * height;
    const uint32_t offset = width * height;
    std::shared_ptr<jcl::OpenCLBufferData> blue =
        buffer->createSubBuffer(nelems, offset);

    // Set the blue channel to pi.
    FillBuffer(context.get(), dev_id, static_cast<float>(M_PI), blue);

    // Now get the larger GPU buffer onto the CPU.
    std::unique_ptr<float[]> buffer_cpu(new float[input_sz]);
    context->readFromBuffer(buffer_cpu.get(), input_sz, dev_id, buffer, true);

    // Now check the output values.
    for (int32_t c = 0; c < nchans; c++) {
      for (int32_t v = 0; v < height; v++) {
        for (int32_t u = 0; u < width; u++) {
          int32_t index = c * width * height + v * width + u;
          if (c == 1) {
            EXPECT_EQ(buffer_cpu[index], static_cast<float>(M_PI));
          } else {
            EXPECT_EQ(buffer_cpu[index], 1.0f);
          }
        }
      }
    }
  }
}
