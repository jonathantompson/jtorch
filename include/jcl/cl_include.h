//
//  cl_include.h
//
//  Internal header (not typically exposed to the outside world).
//

#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#include <assert.h>
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#include <cl.hpp>  // Doesn't exist by default so we have to manually include
#elif defined(_WIN32) || defined (WIN32)
#include <CL/cl.h>
#ifdef CL_VERSION_1_2
#undef CL_VERSION_1_2
#endif
#include <CL/cl.hpp>
#else
#include <CL/cl.h>
#include <cl_1_1.hpp>
#endif

// Like RASSERT(x) we want to be able to assert in debug builds but also print
// meaningful errors (and exit) in release builds.
#define CHECK_ERROR(x) \
  do { \
    const cl_int err_code = (x); \
    if (err_code != CL_SUCCESS) { \
      std::cerr << "CHECK_ERROR - ERROR: " \
                << cl::GetCLErrorEnumString(err_code) << std::endl; \
    } \
    assert(err_code == CL_SUCCESS); \
    if (err_code != CL_SUCCESS) { \
      std::cerr << "CHECK_ERROR: " << __FILE__ << ":" << __LINE__ \
                << std::endl; \
      std::cerr << "Exiting!"; \
      exit(1); \
    }; \
  } while(0);

namespace cl {
std::string GetCLErrorEnumString(const cl_int err_code);
}  // namespace cl

namespace jcl {

typedef enum {
  CLDeviceDefault,
  CLDeviceAll,
  CLDeviceCPU,
  CLDeviceGPU,
  CLDeviceAccelerator
} CLDevice;

typedef enum {
  CLVendorAny,
  CLDeviceNVidia,
  CLDeviceAMD,
  CLDeviceIntel,
} CLVendor;

typedef enum {
  CLBufferTypeReadWrite,
  CLBufferTypeWrite,
  CLBufferTypeRead
} CLBufferType;

}  // namespace jcl

