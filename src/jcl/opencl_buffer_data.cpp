#include <sstream>
#include <iostream>
#include "jcl/opencl_context.h"
#include "jcl/opencl_buffer_data.h"
#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_FREE(x) if (x != NULL) { free(x); x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }

using namespace jcl::data_str;
using std::runtime_error;
using std::cout;
using std::endl;

namespace jcl {

  uint64_t OpenCLBufferData::nelems_allocated_ = 0;
  std::mutex OpenCLBufferData::lock_;

  void OpenCLBufferData::printAllocations(int64_t allocated) {
    float dummy;
    static_cast<void>(dummy);
    if (allocated < 0) {
      std::cout << "\t\tOpenCL deallocation: ";
      allocated *= -1;
    } else {
      std::cout << "\t\tOpenCL allocation: ";
    }
    std::cout << allocated << " elements (global " << 
      ((double)(nelems_allocated_ * sizeof(dummy)) / 1048576.0) << "MB)" << std::endl;
  }

  OpenCLBufferData::OpenCLBufferData(const CLBufferType type, 
    const uint32_t nelems, cl::Context& context) : nelems(nelems), type(type) {
    std::lock_guard<std::mutex> guard(lock_);

    //cl_mem_flags flags = CL_MEM_USE_HOST_PTR;
    cl_mem_flags flags = 0;
    switch (type) {
    case CLBufferTypeRead:
      flags |= CL_MEM_READ_ONLY;
      break;
    case CLBufferTypeWrite:
      flags |= CL_MEM_WRITE_ONLY;
      break;
    case CLBufferTypeReadWrite:
      flags |= CL_MEM_READ_WRITE;
      break;
    default:
      throw runtime_error("OpenCLBufferData::OpenCLBufferData() - "
        "ERROR: Memory type not supported!");
    }

    if (nelems == 0) {
      throw runtime_error("OpenCLBufferData::OpenCLBufferData() - "
        "ERROR: Zero size buffer cannot be allocated!!");
    }

    buffer_.push_back(cl::Buffer(context, flags, sizeof(cl_float) * nelems));
    nelems_allocated_ += nelems;
    reference_count_ = 1;
#if defined(TRACK_ALLOCATIONS)
    printAllocations(nelems);
#endif
  }

  OpenCLBufferData::~OpenCLBufferData() {
    std::lock_guard<std::mutex> guard(lock_);
    size_t sz = buffer_.size();
    buffer_.clear();
    nelems_allocated_ -= nelems;
#if defined(TRACK_ALLOCATIONS)
    if (sz > 0) {
      printAllocations(-((int64_t)nelems));
    }
#endif
  }

  void OpenCLBufferData::addReference() {
    std::lock_guard<std::mutex> guard(lock_);
    if (reference_count_ <= 0 || buffer_.size() <= 0) {
      throw std::runtime_error("Trying to add reference to a released buffer!");
    }
    reference_count_++;
  }

  void OpenCLBufferData::releaseReference() {
    std::lock_guard<std::mutex> guard(lock_);
    if (reference_count_ <= 0 || buffer_.size() <= 0) {
      throw std::runtime_error("Trying to release reference to a released buffer!");
    }
    reference_count_--;
    if (reference_count_ <= 0) {
      buffer_.clear();
      nelems_allocated_ -= nelems;
#if defined(TRACK_ALLOCATIONS)
      printAllocations(-((int64_t)nelems));
#endif
    }
  }

  cl::Buffer& OpenCLBufferData::buffer() {
    std::lock_guard<std::mutex> guard(lock_);
    if (reference_count_ <= 0 || buffer_.size() <= 0) {
      throw std::runtime_error("Trying to access a released buffer!");
    }
    return buffer_[0];
  }

}  // namespace jcl
