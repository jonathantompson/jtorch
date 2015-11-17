#include "jcl/opencl_kernel.h"

#include <iostream>

#include "jcl/opencl_context.h"
#include "jcl/opencl_program.h"

namespace jcl {

  OpenCLKernel::OpenCLKernel(const std::string& kernel_name, 
    OpenCLProgram* program) {
    kernel_name_ = kernel_name;
    program_ = program;  // Ownership is NOT transferred
    compileKernel();
  }

  OpenCLKernel::~OpenCLKernel() {
    // Nothing to do
  }

  void OpenCLKernel::compileKernel() {
    cl_int err;
    kernel_ = cl::Kernel(program_->program(), kernel_name_.c_str(), &err);
    CHECK_ERROR(err);
  }

  void OpenCLKernel::setArg(const uint32_t index, const uint32_t size, 
    void* data) {
    CHECK_ERROR(kernel_.setArg(index, size, data));
  }

}  // namespace jcl
