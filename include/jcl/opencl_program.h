//
//  opencl_program.h
//
//  Created by Jonathan Tompson on 5/12/13.
//
//  Container for storing Program information (char* array and others).
//  This is an internal class and shouldn't be used directly.
//
//  An instance is made PER file (since more than one kernel might be in a any 
//  one source file)
//

#pragma once

#include <string>
#include "jcl/cl_include.h"

namespace jcl {

  struct OpenCLProgram {
  public:
    // This version loads the kernel code from file
    OpenCLProgram(const std::string& kernel_filename, cl::Context& context,
      std::vector<cl::Device>& devices, const bool strict_float);
    // This version copies the kernel code from a c string.  Typical usage is
    // to use the kernel_c_str MD5 as the kernel_name to avoid name clashes.
    OpenCLProgram(const char* kernel_c_str, const std::string& kernel_name,
      cl::Context& context, std::vector<cl::Device>& devices, 
      const bool strict_float);
    ~OpenCLProgram();

    const std::string& filename() { return filename_; }
    cl::Program& program() { return program_; }

  private:
    std::string filename_;
    char* code_;
    cl::Program program_;

    void compileProgram(cl::Context& context,
      std::vector<cl::Device>& devices, const bool strict_float);
    char* readFileToBuffer(const std::string& filename);

    // Non-copyable, non-assignable.
    OpenCLProgram(OpenCLProgram&);
    OpenCLProgram& operator=(const OpenCLProgram&);
  };

};  // namespace jcl
