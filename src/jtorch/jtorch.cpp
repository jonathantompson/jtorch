#include "jtorch/jtorch.h"

#include "jcl/cl_include.h"  // Must come before clBLAS.h
#include <clBLAS.h>
#include <iostream>
#include <mutex>
#include <sstream>

#include "jcl/opencl_context.h"

namespace jtorch {

std::unique_ptr<jcl::OpenCLContext> cl_context = nullptr;
std::mutex cl_context_lock_;
uint32_t deviceid;

void InitJTorch(const bool use_cpu, const uint32_t requested_deviceid,
                const bool verbose_startup) {
  std::lock_guard<std::mutex> lck(cl_context_lock_);
  // Check we haven't already called init.
  RASSERT(cl_context == nullptr);

  if (verbose_startup) {
    std::cout << "Valid OpenCL devices attached:" << std::endl;
    const uint32_t num_devices = jcl::OpenCLContext::printDevices();
    static_cast<void>(num_devices);
  }

  jcl::CLDevice device = use_cpu ? jcl::CLDeviceCPU : jcl::CLDeviceGPU;
  jcl::CLVendor vendor = jcl::CLVendorAny;

  const bool device_exists =
      jcl::OpenCLContext::queryDeviceExists(device, vendor);
  if (!device_exists) {
    if (use_cpu) {
      std::cerr << "No CPU devices attached.";
    } else {
      std::cerr << "No GPU devices attached.";
    }
  }
  RASSERT(device_exists);

  // Otherwise, initialize the context.
  cl_context.reset(new jcl::OpenCLContext());
  cl_context->init(device, jcl::CLVendorAny, verbose_startup);

  // Make sure the user is requesting a device id that exists.
  RASSERT(requested_deviceid < cl_context->getNumDevices());
  deviceid = requested_deviceid;

  std::cout << "Jtorch is using device " << deviceid << ": "
            << cl_context->getDeviceName(deviceid) << std::endl;

  // Startup clblas.
  // TODO(tompson): I have NO idea what device ID this will run on.
  const cl_int blas_ret = clblasSetup();
  const bool blas_ok = (blas_ret == CL_SUCCESS);
  if (!blas_ok) {
    std::cout << "ERROR - InitJTorchInternal: clblasSetup returned error: "
              << jcl::OpenCLContext::getErrorString(blas_ret);
  }
  RASSERT(blas_ok);
}

void ShutdownJTorch() {
  std::lock_guard<std::mutex> lck(cl_context_lock_);
  clblasTeardown();
  cl_context.reset(nullptr);
}

void Sync() {
  std::lock_guard<std::mutex> lck(cl_context_lock_);
  cl_context->sync(deviceid);
}
}  // namespace jtorch
