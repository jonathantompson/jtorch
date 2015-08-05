//
//  cl_include.h
//
//  Internal header (not typically exposed to the outside world).
//

#pragma once

#define __CL_ENABLE_EXCEPTIONS

#if defined(__APPLE__) || defined(__MACOSX)
  #include <OpenCL/opencl.h>
  #include <cl.hpp>  // Doesn't exist by default so we have to manually include it
#else
  #include <CL/cl.hpp>
#endif
