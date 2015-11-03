#include "jcl/opencl_context.h"

#include <iostream>
#include <sstream>

#include "jcl/data_str/hash_funcs.h"
#include "jcl/opencl_buffer_data.h"
#include "jcl/opencl_kernel.h"
#include "jcl/opencl_program.h"

namespace jcl {

std::mutex OpenCLContext::context_lock_;

OpenCLContext::OpenCLContext() {
  cur_program_ = nullptr;
  cur_kernel_ = nullptr;
}

OpenCLContext::~OpenCLContext() {
  // Make sure all queues are empty
  for (uint32_t i = 0; i < queues_.size(); i++) {
    queues_[i].finish();
  }
  queues_.clear();
  devices_.clear();
}

void OpenCLContext::init(const CLDevice device, const CLVendor vendor,
                         const bool verbose_startup) {
  std::lock_guard<std::mutex> lock(context_lock_);

  createContext(device, vendor, verbose_startup);
  InitDevices(device, verbose_startup);
  createCommandQueues(verbose_startup);  // For each device
}

bool OpenCLContext::queryDeviceExists(const CLDevice device,
                                      const CLVendor vendor) {
  std::lock_guard<std::mutex> lock(context_lock_);

  cl::Platform platform;
  bool exists = getPlatform(device, vendor, platform);

  return exists;
}

void OpenCLContext::createContext(const CLDevice device, const CLVendor vendor,
                                  const bool verbose_startup) {
  cl::Platform platform;

  // TODO: We might want to fail more gracefully.
  RASSERT(getPlatform(device, vendor,
                      platform));  // Otherwise no OpenCL platforms were found

  // Use the preferred platform and create a context
  cl_context_properties cps[] = {CL_CONTEXT_PLATFORM,
                                 (cl_context_properties)(platform)(), 0};

  cl_device_type device_cl = CLDevice2CLDeviceType(device);
  cl_int err;
  context_ = cl::Context(device_cl, cps, nullptr, nullptr, &err);
  CHECK_ERROR(err);
  if (verbose_startup) {
    std::cout << "\tCreated OpenCL Context: " << std::endl;
    std::cout << "\t - vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>();
    std::cout << std::endl;
  }
}

void OpenCLContext::InitDevices(const CLDevice device,
                                const bool verbose_startup) {
  cl_int err;
  devices_ = context_.getInfo<CL_CONTEXT_DEVICES>(&err);
  CHECK_ERROR(err);

  // Check that there are devices attadched to the current context
  RASSERT(devices_.size() > 0);

  // Make sure all of the devices match what the user asked for.
  if (device != CLDeviceAll) {
    cl_device_type device_cl = CLDevice2CLDeviceType(device);
    static_cast<void>(device_cl);
    for (uint32_t i = 0; i < devices_.size(); i++) {
      cl_device_type cur_device_cl = devices_[i].getInfo<CL_DEVICE_TYPE>(&err);
      CHECK_ERROR(err);
      RASSERT(cur_device_cl == device_cl);
    }
  }

  for (uint32_t i = 0; i < devices_.size(); i++) {
    if (verbose_startup) {
      std::cout << "\t - device " << i << " CL_DEVICE_NAME: ";
      std::cout << devices_[i].getInfo<CL_DEVICE_NAME>(&err) << std::endl;
      CHECK_ERROR(err);
      std::cout << "\t - device " << i << " CL_DEVICE_VERSION: ";
      std::cout << devices_[i].getInfo<CL_DEVICE_VERSION>(&err) << std::endl;
      CHECK_ERROR(err);
      std::cout << "\t - device " << i << " CL_DRIVER_VERSION: ";
      std::cout << devices_[i].getInfo<CL_DRIVER_VERSION>(&err) << std::endl;
      CHECK_ERROR(err);
      std::cout << "\t - device " << i << " CL_DEVICE_MAX_COMPUTE_UNITS: ";
      std::cout << devices_[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(&err)
                << std::endl;
      CHECK_ERROR(err);
      std::cout << "\t - device " << i
                << " CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: ";
      std::cout << devices_[i].getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>(&err)
                << std::endl;
      CHECK_ERROR(err);
    }
    size_t max_size = devices_[i].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&err);
    CHECK_ERROR(err);
    devices_max_workgroup_size_.push_back((int)max_size);
    if (verbose_startup) {
      std::cout << "\t - device " << i << " CL_DEVICE_MAX_WORK_GROUP_SIZE: ";
      std::cout << devices_max_workgroup_size_[i] << std::endl;
      std::cout << "\t - device " << i << " CL_DEVICE_GLOBAL_MEM_SIZE: ";
      std::cout << devices_[i].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>(&err)
                << std::endl;
      CHECK_ERROR(err);
    }
    std::vector<size_t> item_sizes =
        devices_[i].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>(&err);
    CHECK_ERROR(err);
    std::unique_ptr<uint32_t[]> item_sizes_arr(new uint32_t[3]);
    item_sizes_arr[0] = (uint32_t)item_sizes[0];
    item_sizes_arr[1] = (uint32_t)item_sizes[1];
    item_sizes_arr[2] = (uint32_t)item_sizes[2];
    devices_max_workitem_size_.push_back(std::move(item_sizes_arr));
    if (verbose_startup) {
      std::cout << "\t - device " << i << " CL_DEVICE_MAX_WORK_ITEM_SIZES: ";
      std::cout << "(i=2) ";
      for (int32_t j = 2; j >= 0; j--) {
        std::cout << item_sizes[j] << " ";
      }
      std::cout << "(i=0)";
      std::cout << std::endl;
    }
  }
}

void OpenCLContext::createCommandQueues(const bool verbose_startup) {
  for (uint32_t i = 0; i < devices_.size(); i++) {
    cl_int err;
    queues_.push_back(cl::CommandQueue(context_, devices_[i], 0, &err));
    CHECK_ERROR(err);
  }
}

uint32_t OpenCLContext::getNumDevices() { return (uint32_t)devices_.size(); }

uint32_t OpenCLContext::getMaxWorkgroupSize(const uint32_t i) {
  return devices_max_workgroup_size_[i];
}

uint32_t OpenCLContext::getMaxWorkitemSize(const uint32_t device_index,
                                           const uint32_t dim) {
  return devices_max_workitem_size_[device_index][dim];
}

std::string OpenCLContext::getDeviceName(const uint32_t device_index) {
  std::string name;
  cl_int err;
  name = devices_[device_index].getInfo<CL_DEVICE_NAME>(&err);
  CHECK_ERROR(err);
  return name;
}

CLDevice OpenCLContext::getDeviceType(const uint32_t device_index) {
  cl_device_type type;
  cl_int err;
  type = devices_[device_index].getInfo<CL_DEVICE_TYPE>(&err);
  CHECK_ERROR(err);
  return CLDeviceType2CLDevice(type);
}

uint32_t OpenCLContext::printDevices() {
  // ASSUME THAT THE CONTEXT LOCK HAS ALREADY BEEN TAKEN!
  // Get available platforms
  std::vector<cl::Platform> platforms;
  CHECK_ERROR(cl::Platform::get(&platforms));

  if (platforms.size() == 0) {
    std::cout << "No OpenCL devices found." << std::endl;
    return 0;
  }

  const cl_device_type type_cl = CLDevice2CLDeviceType(CLDeviceAll);

  uint32_t num_devices = 0;
  for (uint32_t i = 0; i < (uint32_t)platforms.size(); i++) {
    std::vector<cl_device_id> devices;
    getPlatformDeviceIDsOfType(platforms[i], type_cl, devices);
    std::cout << "  Platform: " << platforms[i].getInfo<CL_PLATFORM_VENDOR>()
              << " (" << devices.size() << " devices)" << std::endl;
    num_devices += static_cast<uint32_t>(devices.size());

    // Get the devices for this platform.
    std::unique_ptr<char[]> name(new char[1024]);
    for (uint32_t d = 0; d < devices.size(); d++) {
      CHECK_ERROR(clGetDeviceInfo(devices[d], CL_DEVICE_NAME,
                                  1023 * sizeof(name[0]), name.get(), nullptr));
      std::cout << "    Device " << d << " name: " << name.get() << std::endl;
    }
  }
  return num_devices;
}

bool OpenCLContext::getPlatform(const CLDevice device, const CLVendor vendor,
                                cl::Platform& return_platform) {
  // ASSUME THAT THE CONTEXT LOCK HAS ALREADY BEEN TAKEN!
  // Get available platforms
  std::vector<cl::Platform> platforms;
  CHECK_ERROR(cl::Platform::get(&platforms));

  if (platforms.size() == 0) {
    return false;
  }

  const cl_device_type type_cl = CLDevice2CLDeviceType(device);

  int platformID = -1;
  std::string find = CLVendor2String(vendor);
  std::vector<cl_device_id> devices;
  for (uint32_t i = 0; i < (uint32_t)platforms.size(); i++) {
    if (vendor == CLVendorAny ||
        platforms[i].getInfo<CL_PLATFORM_VENDOR>().find(find) !=
            std::string::npos) {
      // Check this platform for the device type we want
      getPlatformDeviceIDsOfType(platforms[i], type_cl, devices);
      if (devices.size() > 0) {
        platformID = i;
        break;
      }
    }
  }

  if (platformID != -1) {
    return_platform = platforms[platformID];
    return true;
  } else {
    return false;
  }
}

void OpenCLContext::getPlatformDeviceIDsOfType(
    cl::Platform& platform, const cl_device_type type,
    std::vector<cl_device_id>& devices) {
  // Unforunately, the ATI C++  API throws a DEEP driver-level memory
  // exception when querying devices on Nvidia hardware.  Use the C API
  // for this task instead.
  if (devices.size() > 0) {
    devices.clear();
  }

  // First, get the size of device list:
  cl_uint num_devices;
  cl_int rc = clGetDeviceIDs(platform(), type, 0, nullptr, &num_devices);
  if (rc != CL_SUCCESS || num_devices == 0) {
    return;
  }

  cl_device_id* dev_ids = new cl_device_id[num_devices];
  CHECK_ERROR(clGetDeviceIDs(platform(), type, num_devices, dev_ids, nullptr));

  for (uint32_t i = 0; i < num_devices; i++) {
    cl_device_type cur_type;
    CHECK_ERROR(clGetDeviceInfo(dev_ids[i], CL_DEVICE_TYPE, sizeof(cur_type),
                                &cur_type, nullptr));
    if (cur_type == type || type == CL_DEVICE_TYPE_ALL) {
      devices.push_back(dev_ids[i]);
    }
  }

  delete[] dev_ids;
}

std::shared_ptr<OpenCLBufferData> OpenCLContext::allocateBuffer(
    const CLBufferType type, const uint32_t nelems) {
  return std::shared_ptr<OpenCLBufferData>(
      new OpenCLBufferData(type, nelems, context_));
}

void OpenCLContext::useKernel(const char* filename, const char* kernel_name,
                              const bool strict_float) {
  // Make sure the program is compiled
  if (cur_program_ == nullptr || cur_program_->filename() != filename) {
    if (programs_.find(filename) == programs_.end()) {
      programs_[filename] = std::unique_ptr<OpenCLProgram>(
          new OpenCLProgram(filename, context_, devices_, strict_float));
    }
    cur_program_ = programs_[filename].get();
  }

  // Make sure the Kernel is compiled
  if (cur_kernel_ == nullptr || cur_kernel_->program() != cur_program_ ||
      cur_kernel_->kernel_name() != kernel_name) {
    std::string id = std::string(filename) + kernel_name;
    if (kernels_.find(id) == kernels_.end()) {
      kernels_[id] = std::unique_ptr<OpenCLKernel>(
          new OpenCLKernel(kernel_name, cur_program_));
    }
    cur_kernel_ = kernels_[id].get();
  }
}

void OpenCLContext::useKernelCStr(const char* kernel_c_str,
                                  const char* kernel_name,
                                  const bool strict_float) {
  // Hash the string and use this for the filename.
  uint32_t hash = jcl::data_str::HashString(
      std::numeric_limits<uint32_t>::max(), kernel_c_str);
  std::stringstream filename;
  filename << "char* kernel. StringHash: " << hash;

  // Make sure the program is compiled
  if (cur_program_ == nullptr || cur_program_->filename() != filename.str()) {
    if (programs_.find(filename.str()) == programs_.end()) {
      programs_[filename.str()] =
          std::unique_ptr<OpenCLProgram>(new OpenCLProgram(
              kernel_c_str, filename.str(), context_, devices_, strict_float));
    }
    cur_program_ = programs_[filename.str()].get();
  }
  // Make sure the Kernel is compiled
  if (cur_kernel_ == nullptr || cur_kernel_->program() != cur_program_ ||
      cur_kernel_->kernel_name() != kernel_name) {
    std::string id = filename.str() + kernel_name;
    if (kernels_.find(id) == kernels_.end()) {
      kernels_[id] = std::unique_ptr<OpenCLKernel>(
          new OpenCLKernel(kernel_name, cur_program_));
    }
    cur_kernel_ = kernels_[id].get();
  }
}

void OpenCLContext::setArg(const uint32_t index,
                           const std::shared_ptr<OpenCLBufferData>& val) {
  // You must call OpenCL::useKernel() first.
  RASSERT(cur_kernel_ != nullptr);
  cur_kernel_->setArg(index, val->buffer());
}

void OpenCLContext::setArg(const uint32_t index, const uint32_t size,
                           void* data) {
  // You must call OpenCL::useKernel() first.
  RASSERT(cur_kernel_ != nullptr);
  cur_kernel_->setArg(index, size, data);
}

void OpenCLContext::sync(const uint32_t device_index) {
  RASSERT(device_index < devices_.size());
  CHECK_ERROR(queues_[device_index].finish());
}

uint32_t OpenCLContext::queryMaxWorkgroupSizeForCurKernel(
    const uint32_t device_index) {
  // You must call OpenCL::useKernel() first.
  RASSERT(cur_kernel_ != nullptr);
  RASSERT(device_index < devices_.size());

  size_t max_workgroup_size;
  cl_int rc = cur_kernel_->kernel().getWorkGroupInfo<size_t>(
      devices_[device_index], CL_KERNEL_WORK_GROUP_SIZE, &max_workgroup_size);
  RASSERT(rc == CL_SUCCESS);
  return (uint32_t)max_workgroup_size;
}

void OpenCLContext::runKernel(const uint32_t device_index, const uint32_t dim,
                              const uint32_t* global_work_size,
                              const uint32_t* local_work_size,
                              const bool blocking) {
  // You must call OpenCL::useKernel() first.
  RASSERT(cur_kernel_ != nullptr);
  RASSERT(device_index < devices_.size());
  RASSERT(dim <= 3);  // OpenCL doesn't support greater than 3 dims!

  uint32_t total_worksize = 1;
  for (uint32_t i = 0; i < dim; i++) {
    // Check that: Global workgroup size is evenly divisible by the local work
    // group size!
    RASSERT((global_work_size[i] % local_work_size[i]) == 0);
    total_worksize *= local_work_size[i];
    // Check that: Local workgroup size is not greater than
    // devices_max_workitem_size_
    RASSERT((uint32_t)local_work_size[i] <=
            (uint32_t)devices_max_workitem_size_[device_index][i]);
  }
  // Check that: Local workgroup size is not greater than
  // CL_DEVICE_MAX_WORK_GROUP_SIZE!
  RASSERT(total_worksize <=
          (uint32_t)devices_max_workgroup_size_[device_index]);
  uint32_t max_size = queryMaxWorkgroupSizeForCurKernel(device_index);
  // Check that: Local workgroup size is not greater than
  // CL_KERNEL_WORK_GROUP_SIZE!
  RASSERT(total_worksize <= (uint32_t)max_size);

  cl::NDRange offset = cl::NullRange;
  cl::NDRange global_work;
  cl::NDRange local_work;
  switch (dim) {
    case 1:
      global_work = cl::NDRange(global_work_size[0]);
      local_work = cl::NDRange(local_work_size[0]);
      break;
    case 2:
      global_work = cl::NDRange(global_work_size[0], global_work_size[1]);
      local_work = cl::NDRange(local_work_size[0], local_work_size[1]);
      break;
    case 3:
      global_work = cl::NDRange(global_work_size[0], global_work_size[1],
                                global_work_size[2]);
      local_work = cl::NDRange(local_work_size[0], local_work_size[1],
                               local_work_size[2]);
      break;
  }
  cl::Event cur_event;
  CHECK_ERROR(queues_[device_index].enqueueNDRangeKernel(
      cur_kernel_->kernel(), offset, global_work, local_work, nullptr,
      &cur_event));

  if (blocking) {
    cur_event.wait();
  }
}

void OpenCLContext::runKernel(const uint32_t device_index, const uint32_t dim,
                              const uint32_t* global_work_size,
                              const bool blocking) {
  // You must call OpenCL::useKernel() first.
  RASSERT(cur_kernel_ != nullptr);
  RASSERT(device_index < devices_.size());
  RASSERT(dim <= 3);  // OpenCL doesn't support greater than 3 dims!

  cl::NDRange offset = cl::NullRange;
  cl::NDRange global_work;
  cl::NDRange local_work = cl::NullRange;  // Let OpenCL Choose
  switch (dim) {
    case 1:
      global_work = cl::NDRange(global_work_size[0]);
      break;
    case 2:
      global_work = cl::NDRange(global_work_size[0], global_work_size[1]);
      break;
    case 3:
      global_work = cl::NDRange(global_work_size[0], global_work_size[1],
                                global_work_size[2]);
      break;
  }
  cl::Event cur_event;
  CHECK_ERROR(queues_[device_index].enqueueNDRangeKernel(
      cur_kernel_->kernel(), offset, global_work, local_work, nullptr,
      &cur_event));

  if (blocking) {
    cur_event.wait();
  }
}

std::string OpenCLContext::CLVendor2String(const CLVendor vendor) {
  std::string str;
  switch (vendor) {
    case CLVendorAny:
      str = "CLVendorAny";
      break;
    case CLDeviceNVidia:
      str = "NVIDIA";
      break;
    case CLDeviceAMD:
      str = "Advanced Micro Devices";
      break;
    case CLDeviceIntel:
      str = "Intel";
      break;
    default:
      std::cout << "Invalid vendor specified" << std::endl;
      RASSERT(false);
      break;
  }
  return str;
}

cl_device_type OpenCLContext::CLDevice2CLDeviceType(const CLDevice device) {
  cl_device_type ret = CL_DEVICE_TYPE_ALL;
  switch (device) {
    case CLDeviceDefault:
      ret = CL_DEVICE_TYPE_DEFAULT;
      break;
    case CLDeviceCPU:
      ret = CL_DEVICE_TYPE_CPU;
      break;
    case CLDeviceGPU:
      ret = CL_DEVICE_TYPE_GPU;
      break;
    case CLDeviceAccelerator:
      ret = CL_DEVICE_TYPE_ACCELERATOR;
      break;
    case CLDeviceAll:
      ret = CL_DEVICE_TYPE_ALL;
      break;
    default:
      std::cout << "Invalid enumerant" << std::endl;
      RASSERT(false);
  }
  return ret;
}

CLDevice OpenCLContext::CLDeviceType2CLDevice(const cl_device_type device) {
  CLDevice ret;
  switch (device) {
    case CL_DEVICE_TYPE_DEFAULT:
      ret = CLDeviceDefault;
      break;
    case CL_DEVICE_TYPE_CPU:
      ret = CLDeviceCPU;
      break;
    case CL_DEVICE_TYPE_GPU:
      ret = CLDeviceGPU;
      break;
    case CL_DEVICE_TYPE_ACCELERATOR:
      ret = CLDeviceAccelerator;
      break;
    case CL_DEVICE_TYPE_ALL:
      ret = CLDeviceAll;
      break;
    default:
      std::cout << "Invalid enumerant" << std::endl;
      RASSERT(false);
      ret = CLDeviceDefault;
      break;
  }
  return ret;
}

std::string OpenCLContext::getErrorString(signed int err) {
  switch (err) {
    case 0:
      return "CL_SUCCESS";
    case -1:
      return "CL_DEVICE_NOT_FOUND";
    case -2:
      return "CL_DEVICE_NOT_AVAILABLE";
    case -3:
      return "CL_COMPILER_NOT_AVAILABLE";
    case -4:
      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5:
      return "CL_OUT_OF_RESOURCES";
    case -6:
      return "CL_OUT_OF_HOST_MEMORY";
    case -7:
      return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8:
      return "CL_MEM_COPY_OVERLAP";
    case -9:
      return "CL_IMAGE_FORMAT_MISMATCH";
    case -10:
      return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11:
      return "CL_BUILD_PROGRAM_FAILURE";
    case -12:
      return "CL_MAP_FAILURE";

    case -30:
      return "CL_INVALID_VALUE";
    case -31:
      return "CL_INVALID_DEVICE_TYPE";
    case -32:
      return "CL_INVALID_PLATFORM";
    case -33:
      return "CL_INVALID_DEVICE";
    case -34:
      return "CL_INVALID_CONTEXT";
    case -35:
      return "CL_INVALID_QUEUE_PROPERTIES";
    case -36:
      return "CL_INVALID_COMMAND_QUEUE";
    case -37:
      return "CL_INVALID_HOST_PTR";
    case -38:
      return "CL_INVALID_MEM_OBJECT";
    case -39:
      return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40:
      return "CL_INVALID_IMAGE_SIZE";
    case -41:
      return "CL_INVALID_SAMPLER";
    case -42:
      return "CL_INVALID_BINARY";
    case -43:
      return "CL_INVALID_BUILD_OPTIONS";
    case -44:
      return "CL_INVALID_PROGRAM";
    case -45:
      return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46:
      return "CL_INVALID_KERNEL_NAME";
    case -47:
      return "CL_INVALID_KERNEL_DEFINITION";
    case -48:
      return "CL_INVALID_KERNEL";
    case -49:
      return "CL_INVALID_ARG_INDEX";
    case -50:
      return "CL_INVALID_ARG_VALUE";
    case -51:
      return "CL_INVALID_ARG_SIZE";
    case -52:
      return "CL_INVALID_KERNEL_ARGS";
    case -53:
      return "CL_INVALID_WORK_DIMENSION";
    case -54:
      return "CL_INVALID_WORK_GROUP_SIZE";
    case -55:
      return "CL_INVALID_WORK_ITEM_SIZE";
    case -56:
      return "CL_INVALID_GLOBAL_OFFSET";
    case -57:
      return "CL_INVALID_EVENT_WAIT_LIST";
    case -58:
      return "CL_INVALID_EVENT";
    case -59:
      return "CL_INVALID_OPERATION";
    case -60:
      return "CL_INVALID_GL_OBJECT";
    case -61:
      return "CL_INVALID_BUFFER_SIZE";
    case -62:
      return "CL_INVALID_MIP_LEVEL";
    case -63:
      return "CL_INVALID_GLOBAL_WORK_SIZE";
    default:
      return "Unknown OpenCL error";
  }
}

std::string OpenCLContext::CLDeviceToString(const CLDevice device) {
  switch (device) {
    case CLDeviceDefault:
      return "CLDeviceDefault";
    case CLDeviceAll:
      return "CLDeviceAll";
    case CLDeviceCPU:
      return "CLDeviceCPU";
    case CLDeviceGPU:
      return "CLDeviceGPU";
    case CLDeviceAccelerator:
      return "CLDeviceDefault";
    default:
      std::cout << "Bad CLDevice" << std::endl;
      RASSERT(false);
      return "Bad CLDevice";
  }
}

std::string OpenCLContext::CLVendorToString(const CLVendor vendor) {
  switch (vendor) {
    case CLVendorAny:
      return "CLVendorAny";
    case CLDeviceNVidia:
      return "CLDeviceNVidia";
    case CLDeviceAMD:
      return "CLDeviceAMD";
    case CLDeviceIntel:
      return "CLDeviceIntel";
    default:
      std::cout << "Bad CLVendor" << std::endl;
      RASSERT(false);
      return "Bad CLVendor";
  }
}

}  // namespace jcl
