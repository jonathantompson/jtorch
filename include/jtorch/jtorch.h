//
//  jtorch.h
//
//  Created by Jonathan Tompson on 5/14/13.
//
//  NOTE: YOU MUST CALL jtorch::InitTorch() before using any of these functions
//  since a valid OpenCL context must exist.  Then you can only use these
//  functions in the same thread that InitTorch was called.
//
//  Call ShutdownJTorch() when finished.
//

#pragma once

#include <memory>
#include <string>

#include "jcl/math/int_types.h"
#include "jcl/opencl_context.h"

#define USE_OPENCL_LOCAL_SIZES  // Let OpenCL choose worksizes

namespace jcl {
class OpenCLContext;
}

namespace jtorch {

// All these functions are thread-safe.
void InitJTorch(const bool use_cpu = false,
                const uint32_t requested_deviceid = 0,
                const bool verbose_startup = true);
void ShutdownJTorch();
void Sync();

// Some constants and globals for the jtorch instance.
// Note: cl_context cannot be a unique_ptr because we need to
// ensure that the context is shutdown last.
// TODO(tompson): This is ugly, put these in a context class.
extern std::unique_ptr<jcl::OpenCLContext> cl_context;
extern std::string jtorch_path;
extern uint32_t deviceid;

};  // namespace jtorch
