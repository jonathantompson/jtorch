//
//  cl_include.h
//
//  Internal header (not typically exposed to the outside world).
//

#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

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

namespace cl {
void CheckError(const cl_int err_code);
std::string GetCLErrorEnumString(const cl_int err_code);
}  // namespace cl

