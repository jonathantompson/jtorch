#include <iostream>
#include "jcl/opencl_program.h"
#include "jcl/jcl.h"
#include "jcl/opencl_context.h"

#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_FREE(x) if (x != NULL) { free(x); x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }

using std::string;
using std::runtime_error;

namespace jcl {

  OpenCLProgram::OpenCLProgram(const std::string& filename, 
    cl::Context& context, std::vector<cl::Device>& devices, 
    const bool strict_float) {
    filename_ = filename;
    code_ = NULL;
    code_ = readFileToBuffer(filename);
    compileProgram(context, devices, strict_float);
  }
  
  OpenCLProgram::OpenCLProgram(const char* kernel_c_str, 
    const std::string& kernel_name, cl::Context& context, 
    std::vector<cl::Device>& devices, const bool strict_float) {
    filename_ = kernel_name;
    code_ = NULL;
    uint32_t str_len = (uint32_t)strlen(kernel_c_str);  // TODO: bounds check
    char dummy_char;
    static_cast<void>(dummy_char);
    code_ = (char*)malloc((str_len + 1) * sizeof(dummy_char));
    strcpy(code_, kernel_c_str);
    compileProgram(context, devices, strict_float);
  }

  OpenCLProgram::~OpenCLProgram() {
    SAFE_FREE(code_);
  }

  // Base code taken from: http://www.opengl.org/wiki/ (tutorial 2)
  char* OpenCLProgram::readFileToBuffer(const std::string& filename) {
    FILE *fptr;
    long length;
    char *buf;

    fptr = fopen(filename.c_str(), "rb");  // Open file for reading
    if (!fptr) {
      string err = string("Renderer::readFileToBuffer() - ERROR: could not"
        " open file (") + filename + ") for reading";
      throw runtime_error(err);
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
    try {
      cl::Program::Sources source(1, std::make_pair(code_, strlen(code_)));
      program_ = cl::Program(context, source);
    } catch (cl::Error err) {
      throw runtime_error(string("cl::Program() failed: ") + 
        OpenCLContext::GetCLErrorString(err));
    }

    try {
      std::cout << "\tBuilding program: " << filename_ << std::endl;
#if !defined(__APPLE__)
      const char* options = "-Werror";  // Make warnings into errors"
#else
      // Unfortunately, on Mac OS X, I think there are warnings that don't get
      // logged, so sometimes kernels wont compile and there's no info to fix it
      const char* options = NULL;
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
      if (options != NULL) {
        std::cout << "\t --> With options: " << options << std::endl;
      }
#endif
      program_.build(devices, options);
#if defined(DEBUG) || defined(_DEBUG)
      std::cout << "\t --> Finished building program" << std::endl;
#endif
    } catch (cl::Error err) {
      std::cout << "\t --> Build failed for source: " << std::endl;
      std::cout << code_ << std::endl;
			string str = program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
      std::cout << "ERROR: program_.build() failed." << std::endl;
      std::cout << "    Program Info: " << str << std::endl;
      throw runtime_error(string("program_.build() failed: ") + 
        OpenCLContext::GetCLErrorString(err));
    }
  }

}  // namespace jcl
