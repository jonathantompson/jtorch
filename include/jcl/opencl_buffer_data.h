//
//  opencl_buffer_data.h
//
//  Created by Jonathan Tompson on 5/12/13.
//
//  class to encapsulate contiguous chunks of OpenCL buffer data.
//

#pragma once

#include <memory>
#include <mutex>

#include "jcl/cl_include.h"

#if defined(DEBUG) || defined(_DEBUG)
#define TRACK_ALLOCATIONS
#endif

namespace jcl {

class OpenCLBufferData {
 public:
  OpenCLBufferData(const CLBufferType type, const uint32_t nelems,
                   cl::Context& context);
  ~OpenCLBufferData();

  // nelems_allocated : total allocation of all active buffers.
  cl::Buffer& buffer();
  const cl::Buffer& buffer() const;
  cl_mem& mem();  // Same as buffer() above, but return the underlining c-object
  uint32_t nelems() const { return nelems_; }
  uint32_t type() const { return type_; }

  // Create a buffer object that is a index into a sub-set of the current
  // buffer. This uses clCreateSubBuffer, which is a driver call, so the
  // function is only as fast as that (which on some machines might not be
  // fast). I *think* it's OK to call this function a lot, but you should
  // definitely test it to make sure the driver isn't leaking memory or
  // behaving badly.
  std::shared_ptr<OpenCLBufferData> createSubBuffer(const uint32_t nelems,
                                                    const uint32_t offset);

 private:
  const uint32_t nelems_;  // ie width * height * feats
  const CLBufferType type_;
  cl::Buffer buffer_;

  static cl_mem_flags getFlagsFromBufferType(CLBufferType type);

  // Private constructor for wrapping an already created cl::Buffer object.
  // This constructor is used only when creating sub-buffers.
  OpenCLBufferData(const CLBufferType type, const uint32_t nelems,
                   cl::Buffer buffer);

  // Non-copyable, non-assignable.
  OpenCLBufferData(const OpenCLBufferData&) = delete;
  OpenCLBufferData& operator=(const OpenCLBufferData&) = delete;
};

};  // namespace jcl
