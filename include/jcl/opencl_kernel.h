//
//  opencl_kernel.h
//
//  Created by Jonathan Tompson on 5/13/13.
//
//  Container for storing Kernel information (char* array and others).
//  This is an internal class and shouldn't be used directly.
//
//  Each kernel is tied to the source file that it comes from (or Program)
//

#pragma once

#include <sstream>
#include <string>
#include "jcl/cl_include.h"
#include "jcl/math/int_types.h"

namespace jcl {

  struct OpenCLProgram;

  struct OpenCLKernel {
  public:
    OpenCLKernel(const std::string& kernel_name, OpenCLProgram* program);
    ~OpenCLKernel();
    
    template <typename T>
    void setArg(const uint32_t index, const T& val);
    void setArg(const uint32_t index, const uint32_t size, void* data);

    const std::string& kernel_name() { return kernel_name_; }
    OpenCLProgram* program() { return program_; }
    cl::Kernel& kernel() { return kernel_; }

  private:
    std::string kernel_name_;
    OpenCLProgram* program_;
    cl::Kernel kernel_;

    void compileKernel();

    // Non-copyable, non-assignable.
    OpenCLKernel(OpenCLKernel&);
    OpenCLKernel& operator=(const OpenCLKernel&);
  };

  template <typename T>
  void OpenCLKernel::setArg(const uint32_t index, const T& val) {
    try {
      kernel_.setArg<T>(index, val);
    } catch (cl::Error err) {
      // This would be nice but on gcc OpenCLContext is undefined and
      // we cannot just include it since it adds circular dependancies.
      // throw std::wruntime_error(std::string("kernel_.setArg() failed: ") +
      //   OpenCLContext::GetCLErrorString(err));
      // Instead let the user figure it out!
      std::stringstream ss;
      ss << "kernel_.setArg() failed: opencl error enum: " << err.err();
      throw std::runtime_error(ss.str());
    }
  }

};  // namespace jcl
