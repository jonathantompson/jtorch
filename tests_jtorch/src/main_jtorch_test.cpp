// THE CPP FUNCTIONALITY HERE IS TO BE TESTED AGAINST "jtorch_test.lua" SCRIPT

#if defined(WIN32) || defined(_WIN32)
#define NOMINMAX
#include <Windows.h>  // Must come first.
#endif

#include <string>

#include "jcl/opencl_context.h"
#include "test_unit/test_unit.h"
#include "test_unit/test_util.h"

// Here we define a pretty messy global.
std::string test_path;

#include "test_tensor.h"
#include "test_modules.h"

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

  if (argc != 2) {
    std::cerr << "Usage: jtorch_test <path_to_jtorch>/tests_jtorch/test_data"
              << std::endl;
    std::cerr << "  (i.e. you must provide the path to the test_data directory)"
              << std::endl;
	std::cerr << "Assuming path is ../tests_jtorch/test_data/" << std::endl;
	test_path = "../../tests_jtorch/test_data/";
  } else {
    test_path = std::string(argv[1]) + std::string("/");
  }

  const bool use_cpu = false;
  const uint32_t dev_id = 0;  // Use the first GPU
  const bool verbose_startup = true;
  jtorch::InitJTorch(use_cpu, dev_id, verbose_startup);

  int test_argc = 1;
  const char* test_argv = "tests_jtorch";

  int ret_val = RUN_TESTS(test_argc, &test_argv);
#ifdef _WIN32
  system("PAUSE");
#endif

  jtorch::ShutdownJTorch();

  return ret_val;
}
