//
//  test_all_tests.cpp
//
//  Created by Jonathan Tompson on 2/23/13.
//
//  Test everything

#include "test_unit/test_unit.h"
#include "test_unit/test_util.h"

// Test basic data structures and utilites.
#include "test_callback.h"
#include "test_callback_queue.h"
#include "test_hash_map_managed.h"
#include "test_pair.h"
#include "test_thread.h"
#include "test_thread_pool.h"
#include "test_vector.h"
#include "test_vector_managed.h"

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
  int ret_val = RUN_TESTS(argc, argv);
#ifdef _WIN32
  system("PAUSE");
#endif
  return ret_val;
}
