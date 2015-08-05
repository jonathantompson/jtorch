//
//  cl_include.h
//
//  Internal header (not typically exposed to the outside world).
//

#pragma once

#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#if defined(__APPLE__) || defined(__MACOSX)
  #include <OpenCL/opencl.h>
  #include <cl.hpp>  // Doesn't exist by default so we have to manually include it
#else
  #include <CL/cl.h>
#ifdef CL_VERSION_1_2
#undef CL_VERSION_1_2
#endif
  #include <CL/cl.hpp>
#endif
