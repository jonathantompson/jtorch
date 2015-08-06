//
//  opencl_context_data.h
//
//  Created by Jonathan Tompson on 5/12/13.
//
//  Struct to store the OpenCL context data and escentially hide the cl.h
//  header from the outside world.  This is an internal class and shouldn't
//  be used directly.
//

#pragma once

#include <memory>
#include <unordered_map>
#include "jcl/cl_include.h"
#include "jcl/jcl.h"
#include "jcl/opencl_kernel.h"
#include "jcl/opencl_buffer_data.h"
#include "jcl/opencl_kernel.h"
#include "jcl/math/math_types.h"  // for jcl::math::Int2 and Int3

#define OPENCL_KERNEL_STARTING_HASH_SIZE 11  // Make it a prime

namespace jcl {

struct OpenCLProgram;

struct OpenCLContext {
 public:
  OpenCLContext();
  ~OpenCLContext();

  cl::Context context;
  std::vector<cl::Device> devices;
  std::vector<cl::CommandQueue> queues;
  
  // Stored by filename, or in the case of char* kernel, it's 32bit hash.
  std::unordered_map<std::string, std::unique_ptr<OpenCLProgram>> programs;
  // Stored by (filename + kernel)
  std::unordered_map<std::string, std::unique_ptr<OpenCLKernel>> kernels;

  std::vector<std::unique_ptr<OpenCLBufferData>> buffers;

  static bool queryDeviceExists(const CLDevice device, const CLVendor vendor);

  void createContext(const CLDevice device, const CLVendor vendor);
  void InitDevices(const CLDevice device);
  void createCommandQueues();

  // Enumerate devices once the context is open
  uint32_t getNumDevices();
  std::string getDeviceName(const uint32_t device_index);
  CLDevice getDeviceType(const uint32_t device_index);
  uint32_t getMaxWorkgroupSize(const uint32_t device_index);
  uint32_t getMaxWorkitemSize(const uint32_t device_index, const uint32_t dim);

  // Memory management
  JCLBuffer allocateBuffer(const CLBufferType type, const uint32_t nelems);
  void addReference(const JCLBuffer buffer);
  void releaseReference(const JCLBuffer buffer);
  template <typename T>
  void writeToBuffer(const T* data, const uint32_t device_index,
                     const JCLBuffer buffer, const bool blocking);
  template <typename T>
  void readFromBuffer(T* data, const uint32_t device_index,
                      const JCLBuffer buffer, const bool blocking);
  static uint64_t nelems_allocated() {
    return OpenCLBufferData::nelems_allocated();
  }
  cl_mem getCLMem(const JCLBuffer buffer);

  // Kernel setup and run
  void useKernel(const char* filename, const char* kernel_name,
                 const bool strict_float);
  void useKernelCStr(const char* kernel_c_str, const char* kernel_name,
                     const bool strict_float);
  void setArg(const uint32_t index, const JCLBuffer& buf);
  template <typename T>
  void setArg(const uint32_t index, const T& val);
  void setArg(const uint32_t index, const uint32_t size, void* data);

  // Run commands to specify the local workgroup size
  void runKernel(const uint32_t device_index, const uint32_t dim,
                 const uint32_t* global_work_size,
                 const uint32_t* local_work_size, const bool blocking);

  // Run commands to let OpenCL choose the local workgroup size
  void runKernel(const uint32_t device_index, const uint32_t dim,
                 const uint32_t* global_work_size, const bool blocking);

  void sync(const uint32_t device_index);  // Blocking until queue is empty

  // devices_max_workgroup_size is the max possible, each kernel might have
  // a specific maximum.  Use this to grab the max size for the compiled
  // kernel.
  uint32_t queryMaxWorkgroupSizeForCurKernel(const uint32_t device_index);

 private:
  OpenCLProgram* cur_program_;  // Not owned here
  OpenCLKernel* cur_kernel_;    // Now owned here
  std::vector<int> devices_max_workgroup_size_;
  std::vector<std::unique_ptr<uint32_t[]>> devices_max_workitem_size_;

  static bool getPlatform(const CLDevice device, const CLVendor vendor,
                          cl::Platform& return_platform);
  static std::string CLVendor2String(const CLVendor vendor);
  static cl_device_type CLDevice2CLDeviceType(const CLDevice device);
  static CLDevice CLDeviceType2CLDevice(const cl_device_type device);
  static void getPlatformDeviceIDsOfType(cl::Platform& platform,
                                         const cl_device_type,
                                         std::vector<cl_device_id>& devices);

  // Non-copyable, non-assignable.
  OpenCLContext(OpenCLContext&);
  OpenCLContext& operator=(const OpenCLContext&);
};

template <typename T>
void OpenCLContext::setArg(const uint32_t index, const T& val) {
  assert(cur_kernel_ != nullptr);  // You must call OpenCL::useKernel() first
  cur_kernel_->setArg(index, val);
}

template <typename T>
void OpenCLContext::writeToBuffer(const T* data, const uint32_t device_index,
                                  const JCLBuffer buffer, const bool blocking) {
  assert(device_index < devices.size());
  assert(static_cast<uint32_t>(buffer) < buffers.size());
  OpenCLBufferData* buf = buffers[(uint32_t)buffer].get();
  cl::Event cur_event;
  cl::CheckError(queues[device_index].enqueueWriteBuffer(
      buf->buffer(), blocking ? CL_TRUE : CL_FALSE, 0,
      buf->nelems * sizeof(data[0]), data, nullptr, &cur_event));
  if (blocking) {
    cur_event.wait();
  }
}

template <typename T>
void OpenCLContext::readFromBuffer(T* data, const uint32_t device_index,
                                   const JCLBuffer buffer,
                                   const bool blocking) {
  assert(device_index < devices.size());
  assert(static_cast<uint32_t>(buffer) < buffers.size());
  OpenCLBufferData* buf = buffers[(uint32_t)buffer].get();
  cl::Event cur_event;
  cl::CheckError(queues[device_index].enqueueReadBuffer(
      buf->buffer(), blocking ? CL_TRUE : CL_FALSE, 0,
      buf->nelems * sizeof(data[0]), data, nullptr, &cur_event));
  if (blocking) {
    cur_event.wait();
  }
}

};  // namespace jcl
