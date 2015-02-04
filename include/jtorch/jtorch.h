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

#include <string>
#include "jcl/math/int_types.h"

#define USE_OPENCL_LOCAL_SIZES  // Let OpenCL choose worksizes

namespace jcl { class JCL; }

namespace jtorch {

  // path_to_jtorch should be the path to "/path/to/jtorch"
  // InitJTorch - Throws exception on multiple init
  void InitJTorch(const std::string& path_to_jtorch, 
    const bool use_cpu = false);  // Thread safe
  // InitJTorchSafe - Multiple init OK
  void InitJTorchSafe(const std::string& path_to_jtorch, 
    const bool use_cpu = false);  // Thread safe
  void ShutdownJTorch();  // Thread safe
  void Sync();  // NOT Thread safe

  // Some constants and globals for the jtorch instance
  extern jcl::JCL* cl_context;
  extern std::string jtorch_path;
  const uint32_t deviceid = 0;

};  // namespace jtorch
