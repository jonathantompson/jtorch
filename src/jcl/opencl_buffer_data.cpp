#include "jcl/opencl_buffer_data.h"

#include <iostream>
#include <sstream>

#include "jcl/opencl_context.h"

namespace jcl {

OpenCLBufferData::OpenCLBufferData(const CLBufferType type,
                                   const uint32_t nelems, cl::Context& context)
    : nelems_(nelems), type_(type) {
  // Zero size buffer cannot be allocated!
  RASSERT(nelems_ > 0);

  cl_mem_flags flags = getFlagsFromBufferType(type_);
  buffer_ = cl::Buffer(context, flags, sizeof(cl_float) * nelems_);
}

// Private constructor.
OpenCLBufferData::OpenCLBufferData(const CLBufferType type,
                                   const uint32_t nelems, cl::Buffer buffer)
    : nelems_(nelems), type_(type), buffer_(buffer) {}

OpenCLBufferData::~OpenCLBufferData() {}

cl::Buffer& OpenCLBufferData::buffer() { return buffer_; }

const cl::Buffer& OpenCLBufferData::buffer() const { return buffer_; }

cl_mem& OpenCLBufferData::mem() { return buffer_(); }

std::shared_ptr<OpenCLBufferData> OpenCLBufferData::createSubBuffer(
    const uint32_t nelems, const uint32_t offset) {
  // Make sure the requested memory fits in the current buffer.
  RASSERT(nelems + offset <= nelems_);

  cl_mem_flags flags = getFlagsFromBufferType(type_);
  cl_buffer_region info;
  info.origin = sizeof(cl_float) * offset;
  info.size = sizeof(cl_float) * nelems;
  cl_int err;
  cl::Buffer new_buffer =
      buffer_.createSubBuffer(flags, CL_BUFFER_CREATE_TYPE_REGION, &info, &err);
  CHECK_ERROR(err);

  return std::shared_ptr<OpenCLBufferData>(
      new OpenCLBufferData(type_, nelems, new_buffer));
}

cl_mem_flags OpenCLBufferData::getFlagsFromBufferType(CLBufferType type) {
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
      std::cout << "OpenCLBufferData::getFlagsFromBufferType() - "
                   "ERROR: Memory type not supported!"
                << std::endl;
      RASSERT(false);
  }
  return flags;
}

}  // namespace jcl
