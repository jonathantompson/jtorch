#include <sstream>
#include <iostream>
#include "jcl/data_str/hash_funcs.h"
#include "jcl/jcl.h"
#include "jcl/opencl_context.h"
#include "jcl/opencl_program.h"
#include "jcl/opencl_kernel.h"
#include "jcl/opencl_buffer_data.h"

namespace jcl {

OpenCLContext::OpenCLContext() {
  cur_program_ = nullptr;
  cur_kernel_ = nullptr;
}

OpenCLContext::~OpenCLContext() {
  // Make sure all queues are empty
  for (uint32_t i = 0; i < queues.size(); i++) {
    queues[i].finish();
  }
  queues.clear();
  devices.clear();
}

bool OpenCLContext::queryDeviceExists(const CLDevice device,
                                      const CLVendor vendor) {
  cl::Platform platform;
  return getPlatform(device, vendor, platform);
}

void OpenCLContext::createContext(const CLDevice device,
                                  const CLVendor vendor) {
  cl::Platform platform;

  // TODO: We might want to fail more gracefully.
  assert(getPlatform(device, vendor,
                     platform));  // Otherwise no OpenCL platforms were found

  // Use the preferred platform and create a context
  cl_context_properties cps[] = {CL_CONTEXT_PLATFORM,
                                 (cl_context_properties)(platform)(), 0};

  cl_device_type device_cl = CLDevice2CLDeviceType(device);
  cl_int err;
  context = cl::Context(device_cl, cps, nullptr, nullptr, &err);
  cl::CheckError(err);
  std::cout << "\tCreated OpenCL Context: " << std::endl;
  std::cout << "\t - vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>();
  std::cout << std::endl;
}

void OpenCLContext::InitDevices(const CLDevice device) {
  cl_int err;
  devices = context.getInfo<CL_CONTEXT_DEVICES>(&err);
  cl::CheckError(err);

  // Check that there are devices attadched to the current context
  assert(devices.size() > 0);

  // Make sure all of the devices match what the user asked for
  cl_device_type device_cl = CLDevice2CLDeviceType(device);
  static_cast<void>(device_cl);
  for (uint32_t i = 0; i < devices.size(); i++) {
    cl_device_type cur_device_cl = devices[i].getInfo<CL_DEVICE_TYPE>(&err);
    static_cast<void>(cur_device_cl);
    cl::CheckError(err);
    assert(cur_device_cl == device_cl);
  }

  for (uint32_t i = 0; i < devices.size(); i++) {
    std::cout << "\t - device " << i << " CL_DEVICE_NAME: ";
    std::cout << devices[i].getInfo<CL_DEVICE_NAME>(&err) << std::endl;
    cl::CheckError(err);
    std::cout << "\t - device " << i << " CL_DEVICE_VERSION: ";
    std::cout << devices[i].getInfo<CL_DEVICE_VERSION>(&err) << std::endl;
    cl::CheckError(err);
    std::cout << "\t - device " << i << " CL_DRIVER_VERSION: ";
    std::cout << devices[i].getInfo<CL_DRIVER_VERSION>(&err) << std::endl;
    cl::CheckError(err);
    std::cout << "\t - device " << i << " CL_DEVICE_MAX_COMPUTE_UNITS: ";
    std::cout << devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(&err) << std::endl;
    cl::CheckError(err);
    std::cout << "\t - device " << i << " CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: ";
    std::cout << devices[i].getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>(&err)
         << std::endl;
    cl::CheckError(err);
    std::cout << "\t - device " << i << " CL_DEVICE_MAX_WORK_GROUP_SIZE: ";
    size_t max_size = devices[i].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&err);
    cl::CheckError(err);
    devices_max_workgroup_size_.push_back((int)max_size);
    std::cout << devices_max_workgroup_size_[i] << std::endl;
    std::cout << "\t - device " << i << " CL_DEVICE_GLOBAL_MEM_SIZE: ";
    std::cout << devices[i].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>(&err) << std::endl;
    cl::CheckError(err);
    std::cout << "\t - device " << i << " CL_DEVICE_MAX_WORK_ITEM_SIZES: ";
    std::vector<size_t> item_sizes =
        devices[i].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>(&err);
    cl::CheckError(err);
    std::unique_ptr<uint32_t[]> item_sizes_arr(new uint32_t[3]);
    item_sizes_arr[0] = (uint32_t)item_sizes[0];
    item_sizes_arr[1] = (uint32_t)item_sizes[1];
    item_sizes_arr[2] = (uint32_t)item_sizes[2];
    devices_max_workitem_size_.push_back(std::move(item_sizes_arr));
    std::cout << "(i=2) ";
    for (int32_t j = 2; j >= 0; j--) {
      std::cout << item_sizes[j] << " ";
    }
    std::cout << "(i=0)";
    std::cout << std::endl;
  }
}

void OpenCLContext::createCommandQueues() {
  for (uint32_t i = 0; i < devices.size(); i++) {
    cl_int err;
    queues.push_back(cl::CommandQueue(context, devices[i], 0, &err));
    cl::CheckError(err);
  }
}

uint32_t OpenCLContext::getNumDevices() { return (uint32_t)devices.size(); }

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
  name = devices[device_index].getInfo<CL_DEVICE_NAME>(&err);
  cl::CheckError(err);
  return name;
}

CLDevice OpenCLContext::getDeviceType(const uint32_t device_index) {
  cl_device_type type;
  cl_int err;
  type = devices[device_index].getInfo<CL_DEVICE_TYPE>(&err);
  cl::CheckError(err);
  return CLDeviceType2CLDevice(type);
}

bool OpenCLContext::getPlatform(const CLDevice device, const CLVendor vendor,
                                cl::Platform& return_platform) {
  // ASSUME THAT THE CONTEXT LOCK HAS ALREADY BEEN TAKEN!
  // Get available platforms
  std::vector<cl::Platform> platforms;
  cl::CheckError(cl::Platform::get(&platforms));

  if (platforms.size() == 0) {
    return false;
  }

  cl_device_type type_cl = CLDevice2CLDeviceType(device);

  int platformID = -1;
  std::string find = CLVendor2String(vendor);
  std::vector<cl_device_id> devices;
  for (uint32_t i = 0; i < (uint32_t)platforms.size(); i++) {
    if (vendor == CLVendorAny ||
        platforms[i].getInfo<CL_PLATFORM_VENDOR>().find(find) != std::string::npos) {
      // Check this platform for the device type we want
      getPlatformDeviceIDsOfType(platforms[i], type_cl, devices);
      if (devices.size() > 0) {
        platformID = i;
        break;
      }
    }
  }

  if (platformID == -1) {
    std::cout << "\tNo compatible OpenCL platform found" << std::endl;
    std::cout << "\tDevices attached to each platform:" << std::endl;
    char* name = new char[1024];
    for (uint32_t i = 0; i < (uint32_t)platforms.size(); i++) {
      std::cout << "\tPlatform " << i << ": ";
      std::cout << platforms[i].getInfo<CL_PLATFORM_VENDOR>() << std::endl;
      getPlatformDeviceIDsOfType(platforms[i], CL_DEVICE_TYPE_ALL, devices);
      for (uint32_t j = 0; j < (uint32_t)devices.size(); j++) {
        cl::CheckError(clGetDeviceInfo(devices[i], CL_DEVICE_NAME,
                                   1023 * sizeof(name[0]), &name, nullptr));
        std::cout << "\t - device " << j << " name: " << name << std::endl;
      }
    }
    delete[] name;
    return false;
  }

  return_platform = platforms[platformID];
  return true;
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
  cl::CheckError(
      clGetDeviceIDs(platform(), type, num_devices, dev_ids, nullptr));

  for (uint32_t i = 0; i < num_devices; i++) {
    cl_device_type cur_type;
    cl::CheckError(clGetDeviceInfo(dev_ids[i], CL_DEVICE_TYPE, sizeof(cur_type),
                                   &cur_type, nullptr));
    if (cur_type == type) {
      devices.push_back(dev_ids[i]);
    }
  }

  delete[] dev_ids;
}

JCLBuffer OpenCLContext::allocateBuffer(const CLBufferType type,
                                        const uint32_t nelems) {
  buffers.push_back(std::unique_ptr<OpenCLBufferData>(new OpenCLBufferData(type, nelems, context)));
  return (JCLBuffer)(buffers.size() - 1);
}

cl_mem OpenCLContext::getCLMem(const JCLBuffer buffer) {
  assert(buffer < buffers.size());
  return buffers[buffer]->buffer()();
}

void OpenCLContext::releaseReference(const JCLBuffer buffer) {
  assert(buffer < buffers.size());
  buffers[buffer]->releaseReference();
}

void OpenCLContext::addReference(const JCLBuffer buffer) {
  assert(buffer < buffers.size());
  buffers[buffer]->addReference();
}

void OpenCLContext::useKernel(const char* filename, const char* kernel_name,
                              const bool strict_float) {
  // Make sure the program is compiled
  if (cur_program_ == nullptr || cur_program_->filename() != filename) {
	  if (programs.find(filename) == programs.end()) {
      programs[filename] = std::unique_ptr<OpenCLProgram>(
        new OpenCLProgram(filename, context, devices, strict_float));
    }
    cur_program_ = programs[filename].get();
  }

  // Make sure the Kernel is compiled
  if (cur_kernel_ == nullptr || cur_kernel_->program() != cur_program_ ||
    cur_kernel_->kernel_name() != kernel_name) {
    std::string id = std::string(filename) + kernel_name;
    if (kernels.find(id) == kernels.end()) {
      kernels[id] = std::unique_ptr<OpenCLKernel>(
        new OpenCLKernel(kernel_name, cur_program_));
    }
    cur_kernel_ = kernels[id].get();
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
     if (programs.find(filename.str()) == programs.end()) {
      programs[filename.str()] = std::unique_ptr<OpenCLProgram>(
        new OpenCLProgram(kernel_c_str, filename.str(), context,
                          devices, strict_float));
    }
    cur_program_ = programs[filename.str()].get();
  }
  // Make sure the Kernel is compiled
  if (cur_kernel_ == nullptr || cur_kernel_->program() != cur_program_ ||
      cur_kernel_->kernel_name() != kernel_name) {
    std::string id = filename.str() + kernel_name;
    if (kernels.find(id) == kernels.end()) {
      kernels[id] = std::unique_ptr<OpenCLKernel>(
        new OpenCLKernel(kernel_name, cur_program_));
    }
    cur_kernel_ = kernels[id].get();
  }
}

void OpenCLContext::setArg(const uint32_t index, const JCLBuffer& val) {
  // You must call OpenCL::useKernel() first.
  assert(cur_kernel_ != nullptr);
  cur_kernel_->setArg(index, buffers[(uint32_t)val]->buffer());
}

void OpenCLContext::setArg(const uint32_t index, const uint32_t size,
                           void* data) {
  // You must call OpenCL::useKernel() first.
  assert(cur_kernel_ != nullptr);
  cur_kernel_->setArg(index, size, data);
}

void OpenCLContext::sync(const uint32_t device_index) {
  assert(device_index < devices.size());
  cl::CheckError(queues[device_index].finish());
}

uint32_t OpenCLContext::queryMaxWorkgroupSizeForCurKernel(
    const uint32_t device_index) {
  // You must call OpenCL::useKernel() first.
  assert(cur_kernel_ != nullptr);
  assert(device_index < devices.size());

  size_t max_workgroup_size;
  cl_int rc = cur_kernel_->kernel().getWorkGroupInfo<size_t>(
      devices[device_index], CL_KERNEL_WORK_GROUP_SIZE, &max_workgroup_size);
  static_cast<void>(rc);
  assert(rc == CL_SUCCESS);
  return (uint32_t)max_workgroup_size;
}

void OpenCLContext::runKernel(const uint32_t device_index, const uint32_t dim,
                              const uint32_t* global_work_size,
                              const uint32_t* local_work_size,
                              const bool blocking) {
  // You must call OpenCL::useKernel() first.
  assert(cur_kernel_ != nullptr);
  assert(device_index < devices.size());
  assert(dim <= 3);  // OpenCL doesn't support greater than 3 dims!

  uint32_t total_worksize = 1;
  for (uint32_t i = 0; i < dim; i++) {
    // Check that: Global workgroup size is evenly divisible by the local work
    // group size!
    assert((global_work_size[i] % local_work_size[i]) == 0);
    total_worksize *= local_work_size[i];
    // Check that: Local workgroup size is not greater than
    // devices_max_workitem_size_
    assert(local_work_size[i] <=
           (int)devices_max_workitem_size_[device_index][i]);
  }
  // Check that: Local workgroup size is not greater than
  // CL_DEVICE_MAX_WORK_GROUP_SIZE!
  assert(total_worksize <= (uint32_t)devices_max_workgroup_size_[device_index]);
  uint32_t max_size = queryMaxWorkgroupSizeForCurKernel(device_index);
  static_cast<void>(max_size);
  // Check that: Local workgroup size is not greater than
  // CL_KERNEL_WORK_GROUP_SIZE!
  assert(total_worksize <= (uint32_t)max_size);

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
  cl::CheckError(queues[device_index].enqueueNDRangeKernel(
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
  assert(cur_kernel_ != nullptr);
  assert(device_index < devices.size());
  assert(dim <= 3);  // OpenCL doesn't support greater than 3 dims!

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
  cl::CheckError(queues[device_index].enqueueNDRangeKernel(
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
      assert(false);
      break;
  }
  return str;
}

cl_device_type OpenCLContext::CLDevice2CLDeviceType(const CLDevice device) {
  cl_device_type ret;
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
      assert(false);
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
      assert(false);
      ret  = CLDeviceDefault;
      break;
  }
  return ret;
}

}  // namespace jcl
