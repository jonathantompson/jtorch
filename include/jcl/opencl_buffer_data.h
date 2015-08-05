//
//  opencl_buffer_data.h
//
//  Created by Jonathan Tompson on 5/12/13.
//
//  Struct to store the OpenCL buffer data.  This is an internal class.  User
//  code should only interact with OpenClBuffer type (which are only handles)
//

#pragma once

#include "jcl/cl_include.h"
#include "jcl/jcl.h"

#if defined(DEBUG) || defined(_DEBUG)
  #define TRACK_ALLOCATIONS
#endif

namespace jcl {

  struct OpenCLBufferData {
  public:
    OpenCLBufferData(const CLBufferType type, const uint32_t nelems,
      cl::Context& context);
    ~OpenCLBufferData();

    const uint32_t nelems;  // ie width * height * feats
    const CLBufferType type;
    
    OpenCLBufferData& operator=(const OpenCLBufferData&);

    void addReference();
    void releaseReference();
    static uint64_t nelems_allocated() { return nelems_allocated_; }
    cl::Buffer& buffer();

  private:
    std::vector<cl::Buffer> buffer_;  // Wrap in vector so we can release on demand
    int32_t reference_count_;
    static uint64_t nelems_allocated_;
    static std::mutex lock_;
    static void printAllocations(int64_t mem_allocated);  // negative on release

    // Non-copyable, non-assignable.
    OpenCLBufferData(OpenCLBufferData&);
  };

};  // namespace jcl
