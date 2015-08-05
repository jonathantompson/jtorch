//
//  test_all_tests.cpp
//
//  Created by Jonathan Tompson on 2/23/13.
//
//  Test everything

#include "test_unit/test_unit.h"
#include "test_unit/test_util.h"

#include "test_callback.h"
#include "test_callback_queue.h"
#include "test_thread.h"
#include "test_thread_pool.h"

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
