//
//  opencl_context_data.h
//
//  Created by Jonathan Tompson on 5/12/13.
//
//  Class to store the OpenCL context data and manage memory.
//
// USAGE:
// using namespace jcl;
// CLDevice device = CLDeviceGPU;
// CLVender vendor = CLVendorAny;
// std::unique_ptr<OpenCLContext> context;
// if (!OpenCLContext::queryDeviceExists(device, vendor) {
//   std::cout << "Error: couldn't find valid devices." << std::endl;
//   std::cout << "  Available devices:" << std::endl;
//   OpenCLContext::PrintDevices();
// } else {
//   context.init(device, vendor);
// }
// std::cout << "Found " << context.getNumDevices() << " devices:" << std::endl;
// for (uint32_t i = 0; i < context.getNumDevices(); i++) {
//   std::cout << i << ": " << context.getDeviceName(i) << std::endl;
// }

#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>

#include "jcl/cl_include.h"
#include "jcl/math/math_types.h"  // for jcl::math::Int2 and Int3
#include "jcl/opencl_buffer_data.h"
#include "jcl/opencl_kernel.h"

// RASSERT is a pretty hacky macro that will assert
// on debug and crash on release. Used all throughout
// jcl and jtorch.
#if defined(WIN32) || defined(_WIN32)
#define PAUSE system("PAUSE")
#else
#define PAUSE 
#endif

#define RASSERT(x) \
  do { \
    const bool result = (x); \
    assert(result); \
    if (!result) { \
      std::cerr << "RASSERT failed: " << __FILE__ << ":" << __LINE__ \
                << std::endl; \
      std::cerr << "Exiting!"; \
      PAUSE; \
      exit(1); \
    }; \
  } while(0);

#define OPENCL_KERNEL_STARTING_HASH_SIZE 11  // Make it a prime

namespace jcl {

class OpenCLBufferData;
class OpenCLProgram;

class OpenCLContext {
 public:
  OpenCLContext();
  ~OpenCLContext();

  void init(const CLDevice device_type, const CLVendor vendor_type,
            const bool verbose_startup);
  static bool queryDeviceExists(const CLDevice device, const CLVendor vendor);
  // printDevices will print to std::cout all platforms and devices, but also
  // it will return the total number of available devices.
  static uint32_t printDevices();

  // Enumerate devices once the context is open
  uint32_t getNumDevices();
  std::string getDeviceName(const uint32_t device_index);

  CLDevice getDeviceType(const uint32_t device_index);
  uint32_t getMaxWorkgroupSize(const uint32_t device_index);
  uint32_t getMaxWorkitemSize(const uint32_t device_index, const uint32_t dim);

  // Memory management
  // Note: according to the OpenCL spec, all buffers are visible to all
  // devices. In a multi-device environment, the actual buffer is allocated on
  // first use, so we don't need to specify a device index when creating the
  // buffer, but we will have to specify it when calling a kernel or reading and
  // writing to the buffer.
  std::shared_ptr<OpenCLBufferData> allocateBuffer(const CLBufferType type,
                                                   const uint32_t nelems);
  template <typename T>
  void writeToBuffer(const T* data, const uint32_t data_sz,
                     const uint32_t device_index,
                     const std::shared_ptr<OpenCLBufferData> buffer,
                     const bool blocking);
  template <typename T>
  void readFromBuffer(T* data, const uint32_t data_sz,
                      const uint32_t device_index,
                      const std::shared_ptr<OpenCLBufferData> buffer,
                      const bool blocking);

  // Kernel setup and run
  void useKernel(const char* filename, const char* kernel_name,
                 const bool strict_float = false);
  void useKernelCStr(const char* kernel_c_str, const char* kernel_name,
                     const bool strict_float = false);
  void setArg(const uint32_t index,
              const std::shared_ptr<OpenCLBufferData>& buf);
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

  // Get a mutable pointer to the command queue.
  cl::CommandQueue* getQueue(uint32_t device_index) {
    return &queues_[device_index];
  }

  // devices_max_workgroup_size is the max possible, each kernel might have
  // a specific maximum.  Use this to grab the max size for the compiled
  // kernel.
  uint32_t queryMaxWorkgroupSizeForCurKernel(const uint32_t device_index);

  static std::string CLDeviceToString(const CLDevice device);
  static std::string CLVendorToString(const CLVendor vendor);

  // Get the cl_int error string
  static std::string getErrorString(const signed int err);

 private:
  cl::Context context_;
  std::vector<cl::Device> devices_;
  std::vector<cl::CommandQueue> queues_;

  // Stored by filename, or in the case of char* kernel, it's 32bit hash.
  std::unordered_map<std::string, std::unique_ptr<OpenCLProgram>> programs_;
  // Stored by (filename + kernel)
  std::unordered_map<std::string, std::unique_ptr<OpenCLKernel>> kernels_;

  static std::mutex context_lock_;
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
  void InitDevices(const CLDevice device, const bool verbose_startup);
  void createCommandQueues(const bool verbose_startup);
  void createContext(const CLDevice device, const CLVendor vendor,
                     const bool verbose_startup);

  // Non-copyable, non-assignable.
  OpenCLContext(const OpenCLContext&) = delete;
  OpenCLContext& operator=(const OpenCLContext&) = delete;
};

template <typename T>
void OpenCLContext::setArg(const uint32_t index, const T& val) {
  RASSERT(cur_kernel_ != nullptr);  // You must call OpenCL::useKernel() first
  cur_kernel_->setArg(index, val);
}

template <typename T>
void OpenCLContext::writeToBuffer(
    const T* data, const uint32_t data_sz, const uint32_t device_index,
    const std::shared_ptr<OpenCLBufferData> buffer, const bool blocking) {
  // This will fail if the data size is not the buffer size.
  RASSERT(data_sz <= buffer->nelems());
  cl::Event cur_event;
  CHECK_ERROR(queues_[device_index].enqueueWriteBuffer(
      buffer->buffer(), blocking ? CL_TRUE : CL_FALSE, 0,
      data_sz * sizeof(data[0]), data, nullptr, &cur_event));
  if (blocking) {
    cur_event.wait();
  }
}

template <typename T>
void OpenCLContext::readFromBuffer(
    T* data, const uint32_t data_sz, const uint32_t device_index,
    const std::shared_ptr<OpenCLBufferData> buffer, const bool blocking) {
  // This will fail if the data size is not the buffer size.
  RASSERT(data_sz <= buffer->nelems());
  cl::Event cur_event;
  CHECK_ERROR(queues_[device_index].enqueueReadBuffer(
      buffer->buffer(), blocking ? CL_TRUE : CL_FALSE, 0,
      data_sz * sizeof(data[0]), data, nullptr, &cur_event));
  if (blocking) {
    cur_event.wait();
  }
}

};  // namespace jcl
