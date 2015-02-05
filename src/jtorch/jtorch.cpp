#include <mutex>
#include <iostream>
#include <sstream>
#include "jcl/jcl.h"
#include "jtorch/jtorch.h"
#include <clBLAS.h>

#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }

namespace jtorch {

  jcl::JCL* cl_context = NULL;
  std::mutex cl_context_lock_;
  std::string jtorch_path;

  void InitJTorchInternal(const std::string& path_to_jtorch, 
    const bool use_cpu) {
    const bool strict_float = false;
    if (!strict_float) {
      std::cout << "\tWARNING: not using strict floats." << std::endl;
    }
    if (use_cpu) {
      cl_context = new jcl::JCL(jcl::CLDeviceCPU, jcl::CLVendorAny),
        strict_float;
    } else {
      if (jcl::JCL::queryDeviceExists(jcl::CLDeviceGPU, jcl::CLVendorAny)) {
        cl_context = new jcl::JCL(jcl::CLDeviceGPU, jcl::CLVendorAny,
          strict_float);
      } else {
        std::cout << "\tWARNING: jtorch is using the CPU!" << std::endl;
        // Fall back to using the CPU (if a valid GPU context doesn't exist)
        cl_context = new jcl::JCL(jcl::CLDeviceCPU, jcl::CLVendorAny,
          strict_float);
      }
    }
    jtorch_path = path_to_jtorch;
    if (jtorch_path.at(jtorch_path.size()-1) != '\\' && 
      jtorch_path.at(jtorch_path.size()-1) != '/') {
      jtorch_path = jtorch_path + '/';
    }

    cl_int err = clblasSetup();
    if (err != CL_SUCCESS) {
      std::stringstream ss;
      ss << "ERROR - InitJTorchInternal: clblasSetup returned error: " <<
        jcl::JCL::getErrorString(err);
      throw std::runtime_error(ss.str());
    }
  }

  void InitJTorch(const std::string& path_to_jtorch, const bool use_cpu) {
    std::lock_guard<std::mutex> lck(cl_context_lock_);
    if (cl_context != NULL) {
      throw std::runtime_error("jtorch::InitJTorch() - ERROR: Init called "
        "twice!");
    }
    InitJTorchInternal(path_to_jtorch, use_cpu);
  }

  void InitJTorchSafe(const std::string& path_to_jtorch, const bool use_cpu) {
    std::lock_guard<std::mutex> lck(cl_context_lock_);
    if (cl_context != NULL) {
      return;
    }
    InitJTorchInternal(path_to_jtorch, use_cpu);
  }

  void ShutdownJTorch() {
    std::lock_guard<std::mutex> lck(cl_context_lock_);
    clblasTeardown();
    SAFE_DELETE(cl_context);
  }

  void Sync() {
    cl_context->sync(deviceid);
  }

}  // namespace jtorch
