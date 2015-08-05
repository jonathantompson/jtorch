#include <stdlib.h>
#include <iostream>
#include <sys/types.h>
#include <stdio.h>
#include "clk\clk.h"
#include "debug_util.h"

/* Include the clBLAS header. It includes the appropriate OpenCL headers */
#include <clBLAS.h>

/* This example uses predefined matrices and their characteristics for
* simplicity purpose.
*/

const int kM = 4;
const int kN = 3;
const int kK = 5;

static const cl_float alpha = 10;

static const cl_float A[kM * kK] = {
  11, 12, 13, 14, 15,
  21, 22, 23, 24, 25,
  31, 32, 33, 34, 35,
  41, 42, 43, 44, 45,
};
static const size_t lda = kK;        /* i.e. lda = K */

static const cl_float B[kK * kN] = {
  11, 12, 13,
  21, 22, 23,
  31, 32, 33,
  41, 42, 43,
  51, 52, 53,
};
static const size_t ldb = kN;        /* i.e. ldb = N */

static const cl_float beta = 20;

static cl_float C[kM * kN] = {
  11, 12, 13,
  21, 22, 23,
  31, 32, 33,
  41, 42, 43, 
};
static const size_t ldc = kN;        /* i.e. ldc = N */

static cl_float result[kM * kN];

const char * getErrorString(cl_int err) {
  switch (err) {
  case 0: return "CL_SUCCESS";
  case -1: return "CL_DEVICE_NOT_FOUND";
  case -2: return "CL_DEVICE_NOT_AVAILABLE";
  case -3: return "CL_COMPILER_NOT_AVAILABLE";
  case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case -5: return "CL_OUT_OF_RESOURCES";
  case -6: return "CL_OUT_OF_HOST_MEMORY";
  case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case -8: return "CL_MEM_COPY_OVERLAP";
  case -9: return "CL_IMAGE_FORMAT_MISMATCH";
  case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case -11: return "CL_BUILD_PROGRAM_FAILURE";
  case -12: return "CL_MAP_FAILURE";

  case -30: return "CL_INVALID_VALUE";
  case -31: return "CL_INVALID_DEVICE_TYPE";
  case -32: return "CL_INVALID_PLATFORM";
  case -33: return "CL_INVALID_DEVICE";
  case -34: return "CL_INVALID_CONTEXT";
  case -35: return "CL_INVALID_QUEUE_PROPERTIES";
  case -36: return "CL_INVALID_COMMAND_QUEUE";
  case -37: return "CL_INVALID_HOST_PTR";
  case -38: return "CL_INVALID_MEM_OBJECT";
  case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case -40: return "CL_INVALID_IMAGE_SIZE";
  case -41: return "CL_INVALID_SAMPLER";
  case -42: return "CL_INVALID_BINARY";
  case -43: return "CL_INVALID_BUILD_OPTIONS";
  case -44: return "CL_INVALID_PROGRAM";
  case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
  case -46: return "CL_INVALID_KERNEL_NAME";
  case -47: return "CL_INVALID_KERNEL_DEFINITION";
  case -48: return "CL_INVALID_KERNEL";
  case -49: return "CL_INVALID_ARG_INDEX";
  case -50: return "CL_INVALID_ARG_VALUE";
  case -51: return "CL_INVALID_ARG_SIZE";
  case -52: return "CL_INVALID_KERNEL_ARGS";
  case -53: return "CL_INVALID_WORK_DIMENSION";
  case -54: return "CL_INVALID_WORK_GROUP_SIZE";
  case -55: return "CL_INVALID_WORK_ITEM_SIZE";
  case -56: return "CL_INVALID_GLOBAL_OFFSET";
  case -57: return "CL_INVALID_EVENT_WAIT_LIST";
  case -58: return "CL_INVALID_EVENT";
  case -59: return "CL_INVALID_OPERATION";
  case -60: return "CL_INVALID_GL_OBJECT";
  case -61: return "CL_INVALID_BUFFER_SIZE";
  case -62: return "CL_INVALID_MIP_LEVEL";
  case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
  default: return "Unknown OpenCL error";
  }
}

cl_int CheckError(cl_int err) {
  if (err == CL_SUCCESS) {
    return err;
  }
  std::cout << "Error returned: " << getErrorString(err) << std::endl;
#if defined(WIN32) || defined(_WIN32)
  std::cout << std::endl;
  system("PAUSE");
#endif
  exit(-1);
}

int main( void ) {
#if defined(_DEBUG) || defined(DEBUG)
  jcl::debug::EnableMemoryLeakChecks();
  // jcl::debug::EnableAggressiveMemoryLeakChecks();
  jcl::debug::SetBreakPointOnAlocation(374);
#endif

  cl_int err;

  cl_platform_id platform = 0;
  cl_device_id device = nullptr;
  cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
  cl_context ctx = 0;
  cl_command_queue queue = 0;
  cl_mem bufA, bufB, bufC;
  cl_event event = nullptr;
  int ret = 0;

  /* Setup OpenCL environment. */
  cl_uint num_platforms;
  err = clGetPlatformIDs( 1, nullptr, &num_platforms );
  cl_platform_id* platforms = new cl_platform_id[num_platforms];
  CheckError(err);
  err = clGetPlatformIDs( num_platforms, platforms, nullptr );
  CheckError(err);
  for (cl_uint i = 0; i < num_platforms && device == nullptr; i++) {
    err = clGetDeviceIDs( platforms[i], CL_DEVICE_TYPE_GPU, 1, &device, nullptr );
    if (err == CL_SUCCESS && device != nullptr) {
      platform = platforms[i];
      break;
    }
  }
  CheckError(err);

  props[1] = (cl_context_properties)platform;
  delete[] platforms;
  ctx = clCreateContext( props, 1, &device, nullptr, nullptr, &err );
  CheckError(err);
  queue = clCreateCommandQueue( ctx, device, 0, &err );
  CheckError(err);

  /* Setup clBLAS */
  err = clblasSetup( );
  CheckError(err);

  /* Prepare OpenCL memory objects and place matrices inside them. */
  bufA = clCreateBuffer( ctx, CL_MEM_READ_ONLY, kM * kK * sizeof(*A),
    nullptr, &err );
  CheckError(err);
  bufB = clCreateBuffer( ctx, CL_MEM_READ_ONLY, kK * kN * sizeof(*B),
    nullptr, &err );
  CheckError(err);
  bufC = clCreateBuffer( ctx, CL_MEM_READ_WRITE, kM * kN * sizeof(*C),
    nullptr, &err );
  CheckError(err);

  err = clEnqueueWriteBuffer( queue, bufA, CL_TRUE, 0,
    kM * kK * sizeof( *A ), A, 0, nullptr, nullptr );
  err = clEnqueueWriteBuffer( queue, bufB, CL_TRUE, 0,
    kK * kN * sizeof( *B ), B, 0, nullptr, nullptr );
  err = clEnqueueWriteBuffer( queue, bufC, CL_TRUE, 0,
    kM * kN * sizeof( *C ), C, 0, nullptr, nullptr );

  const double t_test = 5.0;
  std::cout << "Profiling clBLAS for " << t_test << " seconds" << std::endl;
  clk::Clk clk;
  const double t_start = clk.getTime();
  double t_end = t_start;
  uint64_t niters = 0;

  while (t_end - t_start < t_test) {
    /* Call clBLAS extended function. Perform gemm for the lower right sub-matrices */
    err = clblasSgemm( clblasRowMajor, clblasNoTrans, clblasNoTrans, 
      kM, kN, kK,
      alpha, bufA, 0, lda,
      bufB, 0, ldb, beta,
      bufC, 0, ldc,
      1, &queue, 0, nullptr, &event );
    CheckError(err);

    /* Wait for calculations to be finished. */
    err = clWaitForEvents( 1, &event );
    CheckError(err);
    err = clFlush(queue);
    CheckError(err);
    niters++;
    t_end = clk.getTime();
  }

  std::cout << "\t\tExecution time: " << (t_end - t_start) / (double)niters
    << " seconds per sgemm call" << std::endl;

  /* Fetch results of calculations from GPU memory. */
  err = clEnqueueReadBuffer( queue, bufC, CL_TRUE, 0,
    kM * kN * sizeof(*result),
    result, 0, nullptr, nullptr );
  CheckError(err);

  /* Release OpenCL memory objects. */
  clReleaseMemObject( bufC );
  clReleaseMemObject( bufB );
  clReleaseMemObject( bufA );

  /* Finalize work with clBLAS */
  clblasTeardown( );

  /* Release OpenCL working objects. */
  clReleaseCommandQueue( queue );
  clReleaseContext( ctx );

#if defined(WIN32) || defined(_WIN32)
  std::cout << "The test ran OK." << std::endl << std::endl;
  system("PAUSE");
#endif

  return ret;
}