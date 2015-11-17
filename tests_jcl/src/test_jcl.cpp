//
//  test_jcl.cpp
//
//  Created by Jonathan Tompson on 2/23/13.
//
//  Test everything

#if defined(WIN32) || defined(_WIN32)
#define NOMINMAX
#include <Windows.h>  // Must come first.
#endif

#include "test_unit/test_unit.h"
#include "test_unit/test_util.h"

#include "jcl/opencl_context.h"
#include "test_callback.h"
#include "test_callback_queue.h"
#include "test_thread.h"
#include "test_thread_pool.h"

// Test some basic memory handling.
#include "test_memory.h"

// Test a OpenCL kernel on CPU and GPU.
#include "test_convolution.h"

#include "debug_util.h"  // Must come last in .cpp with main

using std::cout;
using std::endl;

int main(int argc, char* argv[]) {
#if defined(_DEBUG) || defined(DEBUG)
  jcl::debug::EnableMemoryLeakChecks();
  #if defined(WIN32) || defined(_WIN32)
  jcl::debug::EnableAggressiveMemoryLeakChecks();
  #endif
  // jcl::debug::SetBreakPointOnAlocation(2546);
#endif

  std::cout << "Valid OpenCL devices attached:" << std::endl;
  const uint32_t num_devices = jcl::OpenCLContext::printDevices();
  if (num_devices < 1) {
    std::cout << "No devices to test. Exiting..." << std::endl;
    exit(-1);
  }

  int ret_val = RUN_TESTS(argc, argv);
#ifdef _WIN32
  system("PAUSE");
#endif
  return ret_val;
}
