#include <assert.h>
#include <iostream>
#include <mutex>
#include <sstream>
#include "jcl/jcl.h"
#include "jtorch/jtorch.h"
#include <clBLAS.h>

namespace jtorch {

  std::unique_ptr<jcl::JCL> cl_context = nullptr;
  std::mutex cl_context_lock_;
  std::string jtorch_path;

  void InitJTorchInternal(const std::string& path_to_jtorch, 
    const bool use_cpu) {
    const bool strict_float = false;
    if (!strict_float) {
      std::cout << "\tWARNING: not using strict floats." << std::endl;
    }
    if (use_cpu) {
      cl_context.reset(new jcl::JCL(jcl::CLDeviceCPU, jcl::CLVendorAny,
        strict_float));
    } else {
      if (jcl::JCL::queryDeviceExists(jcl::CLDeviceGPU, jcl::CLVendorAny)) {
        cl_context.reset(new jcl::JCL(jcl::CLDeviceGPU, jcl::CLVendorAny,
          strict_float));
      } else {
        std::cout << "\tWARNING: jtorch is using the CPU!" << std::endl;
        // Fall back to using the CPU (if a valid GPU context doesn't exist)
        cl_context.reset(new jcl::JCL(jcl::CLDeviceCPU, jcl::CLVendorAny,
          strict_float));
      }
    }
    jtorch_path = path_to_jtorch;
    if (jtorch_path.at(jtorch_path.size()-1) != '\\' && 
      jtorch_path.at(jtorch_path.size()-1) != '/') {
      jtorch_path = jtorch_path + '/';
    }

    cl_int err = clblasSetup();
    if (err != CL_SUCCESS) {
      std::cout << "ERROR - InitJTorchInternal: clblasSetup returned error: " <<
        jcl::JCL::getErrorString(err);
      assert(false);
    }
  }

  void InitJTorch(const std::string& path_to_jtorch, const bool use_cpu) {
    std::lock_guard<std::mutex> lck(cl_context_lock_);
    // Check we haven't already called init.
    assert(cl_context == nullptr);
    InitJTorchInternal(path_to_jtorch, use_cpu);
  }

  void InitJTorchSafe(const std::string& path_to_jtorch, const bool use_cpu) {
    std::lock_guard<std::mutex> lck(cl_context_lock_);
    if (cl_context != nullptr) {
      return;
    }
    InitJTorchInternal(path_to_jtorch, use_cpu);
  }

  void ShutdownJTorch() {
    std::lock_guard<std::mutex> lck(cl_context_lock_);
    clblasTeardown();
    cl_context.reset(nullptr);
  }

  void Sync() {
    cl_context->sync(deviceid);
  }

}  // namespace jtorch
