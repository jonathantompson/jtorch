#include <iostream>
#include "jcl/opencl_program.h"
#include "jcl/jcl.h"
#include "jcl/opencl_context.h"

namespace jcl {

  OpenCLProgram::OpenCLProgram(const std::string& filename, 
    cl::Context& context, std::vector<cl::Device>& devices, 
    const bool strict_float) {
    filename_ = filename;
    code_ = nullptr;
    code_.reset(readFileToBuffer(filename));
    compileProgram(context, devices, strict_float);
  }
  
  OpenCLProgram::OpenCLProgram(const char* kernel_c_str, 
    const std::string& kernel_name, cl::Context& context, 
    std::vector<cl::Device>& devices, const bool strict_float) {
    filename_ = kernel_name;
    code_ = nullptr;
    uint32_t str_len = (uint32_t)strlen(kernel_c_str);  // TODO: bounds check
    code_.reset(new char[str_len + 1]);
    strcpy(code_.get(), kernel_c_str);
    compileProgram(context, devices, strict_float);
  }

  OpenCLProgram::~OpenCLProgram() {
  }

  char* OpenCLProgram::readFileToBuffer(const std::string& filename) {
    FILE *fptr;
    long length;
    char *buf;

    fptr = fopen(filename.c_str(), "rb");  // Open file for reading
    if (!fptr) {
      std::cout << "Renderer::readFileToBuffer() - ERROR: could not"
        " open file (" << filename << ") for reading" << std::endl;
      assert(false);
    }
    fseek(fptr, 0, SEEK_END);  // Seek to the end of the file
    length = ftell(fptr);  // Find out how many bytes into the file we are
    buf = (char*)malloc(length+2);  // Allocate a buffer for the entire length 
                                    // of the file and a null terminator
    fseek(fptr, 0, SEEK_SET);  // Go back to the beginning of the file
    fread(buf, length, 1, fptr);  // Read the contents of the file in to the
                                  // buffer
    fclose(fptr);  // Close the file
    buf[length] = '\n'; 
    buf[length+1] = 0; // Null terminator

    return buf;
  }

  void OpenCLProgram::compileProgram(cl::Context& context,
    std::vector<cl::Device>& devices, const bool strict_float) {
    cl::Program::Sources source(
        1, std::make_pair(code_.get(), strlen(code_.get())));
    cl_int err;
    program_ = cl::Program(context, source, &err);
    cl::CheckError(err);

    std::cout << "\tBuilding program: " << filename_ << std::endl;
#if !defined(__APPLE__)
    const char* options = "-Werror";  // Make warnings into errors"
#else
      // Unfortunately, on Mac OS X, I think there are warnings that don't get
      // logged, so sometimes kernels wont compile and there's no info to fix it
      const char* options = nullptr;
#endif
    if (!strict_float) {
#if !defined(__APPLE__)
      options = "-Werror -cl-mad-enable -cl-fast-relaxed-math "
        "-cl-no-signed-zeros -cl-denorms-are-zero "
        "-cl-unsafe-math-optimizations";
#else
      options = "-cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros "
                "-cl-auto-vectorize-enable -cl-denorms-are-zero "
                "-cl-unsafe-math-optimizations";
#endif
    }
#if defined(DEBUG) || defined(_DEBUG)
    if (options != nullptr) {
      std::cout << "\t --> With options: " << options << std::endl;
    }
#endif
    err = program_.build(devices, options);
#if defined(DEBUG) || defined(_DEBUG)
      std::cout << "\t --> Finished building program" << std::endl;
#endif
    if (err != CL_SUCCESS) {
      std::cout << "\t --> Build failed for source: " << std::endl;
      std::cout << code_.get() << std::endl;
			std::string str = program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
      std::cout << "ERROR: program_.build() failed." 
                << cl::GetCLErrorEnumString(err) << std::endl;
      std::cout << "    Program Info: " << str << std::endl;
      cl::CheckError(err);
    }
  }

}  // namespace jcl
