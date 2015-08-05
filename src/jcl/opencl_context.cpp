#include <sstream>
#include <iostream>
#include "jcl/data_str/hash_funcs.h"
#include "jcl/jcl.h"
#include "jcl/opencl_context.h"
#include "jcl/opencl_program.h"
#include "jcl/opencl_kernel.h"
#include "jcl/opencl_buffer_data.h"
extern "C" {
#include "jcl/string_util/md5_jcl.h"
}

#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_FREE(x) if (x != NULL) { free(x); x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }

using namespace jcl::data_str;
using namespace jcl::math;
using std::runtime_error;
using std::cout;
using std::endl;
using std::string;

namespace jcl {

  OpenCLContext::OpenCLContext() {
    cur_program_ = NULL;
    cur_kernel_ = NULL;
    programs = new HashMapManaged<std::string, OpenCLProgram*>(
      OPENCL_KERNEL_STARTING_HASH_SIZE, &HashString);
    kernels =  new HashMapManaged<std::string, OpenCLKernel*>(
      OPENCL_KERNEL_STARTING_HASH_SIZE, &HashString);
    buffers = new VectorManaged<OpenCLBufferData*>();
  }

  OpenCLContext::~OpenCLContext() {
    // Make sure all queues are empty
    for (uint32_t i = 0; i < queues.size(); i++) {
      queues[i].finish();
    }
    SAFE_DELETE(programs);
    SAFE_DELETE(kernels);
    SAFE_DELETE(buffers);
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
    if ( !getPlatform(device, vendor, platform) ) {
      throw runtime_error("No OpenCL platforms were found");
    }

    // Use the preferred platform and create a context
    cl_context_properties cps[] = { CL_CONTEXT_PLATFORM, 
      (cl_context_properties)(platform)(), 0 };

    cl_device_type device_cl = CLDevice2CLDeviceType(device);
    try {
      context = cl::Context(device_cl, cps);
    } catch (cl::Error err) {
      throw runtime_error(string("cl::Context() failed: ") + 
        GetCLErrorString(err));
    }
    cout << "\tCreated OpenCL Context: " << endl;
    cout << "\t - vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>();
    cout << endl;
  }

  void OpenCLContext::InitDevices(const CLDevice device) {
    try {
      devices = context.getInfo<CL_CONTEXT_DEVICES>();
    } catch (cl::Error err) {
      throw runtime_error(string("context.getInfo() failed: ") + 
        GetCLErrorString(err));
    }

    if (devices.size() <= 0) {
      throw runtime_error("OpenCLContext::getDevice() - ERROR: "
        "No devices are attached to the current context!");
    }

    // Make sure all of the devices match what the user asked for
    cl_device_type device_cl = CLDevice2CLDeviceType(device);
    for (uint32_t i = 0; i < devices.size(); i++) {
      try {
        if (devices[i].getInfo<CL_DEVICE_TYPE>() != device_cl) {
          throw runtime_error("OpenCLContext::InitDevices() - INTERNAL ERROR:"
            " Incorrect device type found!");
        }
      } catch (cl::Error err) {
        throw runtime_error(string("devices[cur_device].getInfo() failed: ") + 
          GetCLErrorString(err));
      }
    }

    try {
      for (uint32_t i = 0; i < devices.size(); i++) {
        cout << "\t - device " << i << " CL_DEVICE_NAME: ";
        cout << devices[i].getInfo<CL_DEVICE_NAME>() << endl;
        cout << "\t - device " << i << " CL_DEVICE_VERSION: ";
        cout << devices[i].getInfo<CL_DEVICE_VERSION>() << endl;
        cout << "\t - device " << i << " CL_DRIVER_VERSION: ";
        cout << devices[i].getInfo<CL_DRIVER_VERSION>() << endl;
        cout << "\t - device " << i << " CL_DEVICE_MAX_COMPUTE_UNITS: ";
        cout << devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
        cout << "\t - device " << i << " CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: ";
        cout << devices[i].getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() <<
                endl;
        cout << "\t - device " << i << " CL_DEVICE_MAX_WORK_GROUP_SIZE: ";
        size_t max_size = devices[i].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
        devices_max_workgroup_size_.pushBack((int)max_size);
        cout << devices_max_workgroup_size_[i] << endl;
        cout << "\t - device " << i << " CL_DEVICE_GLOBAL_MEM_SIZE: ";
        cout << devices[i].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << endl;
        cout << "\t - device " << i << " CL_DEVICE_MAX_WORK_ITEM_SIZES: ";
        std::vector<size_t> item_sizes =
                devices[i].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
        uint32_t* item_sizes_arr = new uint32_t[3];
        item_sizes_arr[0] = (uint32_t)item_sizes[0];
        item_sizes_arr[1] = (uint32_t)item_sizes[1];
        item_sizes_arr[2] = (uint32_t)item_sizes[2];
        // Note: The c_arr ownership is transferred to VectorManaged
        devices_max_workitem_size_.pushBack(item_sizes_arr);
        std::cout << "(i=2) ";
        for (int32_t j = 2; j >= 0; j--) {
          std::cout << item_sizes[j] << " ";
        }
        std::cout << "(i=0)";
        cout << endl;

      }
    } catch (cl::Error err) {
      throw runtime_error(string("devices[cur_device].getInfo() failed: ") + 
        GetCLErrorString(err));
    }
  }

  void OpenCLContext::createCommandQueues() {
    try {
      for (uint32_t i = 0; i < devices.size(); i++) {
        queues.push_back(cl::CommandQueue(context, devices[i]));
      }
    } catch (cl::Error err) {
      throw runtime_error(string("cl::CommandQueue() failed: ") + 
        GetCLErrorString(err));
    }
  }

  uint32_t OpenCLContext::getNumDevices() {
    return (uint32_t)devices.size();
  }

  uint32_t OpenCLContext::getMaxWorkgroupSize(const uint32_t i) {
    return devices_max_workgroup_size_[i];
  }

  uint32_t OpenCLContext::getMaxWorkitemSize(const uint32_t device_index, 
    const uint32_t dim) {
    return devices_max_workitem_size_[device_index][dim];
  }

  std::string OpenCLContext::getDeviceName(const uint32_t device_index) {
    string name;
    try {
      name = devices[device_index].getInfo<CL_DEVICE_NAME>();
    } catch (cl::Error err) {
      throw runtime_error(string("devices[]::getInfo() failed: ") + 
        GetCLErrorString(err));
    }
    return name;
  }

  CLDevice OpenCLContext::getDeviceType(const uint32_t device_index) {
    cl_device_type type;
    try {
      type = devices[device_index].getInfo<CL_DEVICE_TYPE>();
    } catch (cl::Error err) {
      throw runtime_error(string("devices[]::getInfo() failed: ") + 
        GetCLErrorString(err));
    }
    return CLDeviceType2CLDevice(type);
  }

  bool OpenCLContext::getPlatform(const CLDevice device, 
    const CLVendor vendor, cl::Platform& return_platform) {
    // ASSUME THAT THE CONTEXT LOCK HAS ALREADY BEEN TAKEN!
    // Get available platforms
    try {
      std::vector<cl::Platform> platforms;
      cl::Platform::get(&platforms);

      if(platforms.size() == 0) {
        return false;
      }

      cl_device_type type_cl = CLDevice2CLDeviceType(device);

      int platformID = -1;
      std::string find = CLVendor2String(vendor);
      std::vector<cl_device_id> devices;
      for(uint32_t i = 0; i < (uint32_t)platforms.size(); i++) {
        if (vendor == CLVendorAny ||
                platforms[i].getInfo<CL_PLATFORM_VENDOR>().find(find) != string::npos) {
          // Check this platform for the device type we want
          getPlatformDeviceIDsOfType(platforms[i], type_cl, devices);
          if (devices.size() > 0) {
            platformID = i;
            break;
          }
        }
      }
      
      if(platformID == -1) {
        cout << "\tNo compatible OpenCL platform found" << endl;
        cout << "\tDevices attached to each platform:" << endl;
        char* name = new char[1024];
        for (uint32_t i = 0; i < (uint32_t)platforms.size(); i++) {
          cout << "\tPlatform " << i << ": ";
          cout << platforms[i].getInfo<CL_PLATFORM_VENDOR>() << endl;
          getPlatformDeviceIDsOfType(platforms[i], CL_DEVICE_TYPE_ALL, 
            devices);
          for (uint32_t j = 0; j < (uint32_t)devices.size(); j++) {
            CheckError(clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 
              1023 * sizeof(name[0]), &name, NULL));
            cout << "\t - device " << j << " name: " << name << endl;
          }
        }
        delete[] name;
        return false;
      }

      return_platform = platforms[platformID];
      return true;
    } catch (cl::Error err) {
      throw runtime_error(string("Platform query failed: ") + 
        GetCLErrorString(err));
    }
  }

  void OpenCLContext::CheckError(const cl_int err_code) {
#if defined(DEBUG) || defined(_DEBUG)
    if (err_code != CL_SUCCESS) {
      std::stringstream ss;
      ss << "OpenCLContext::CheckError() - ERROR: ";
      ss << GetCLErrorString(err_code);
      throw std::runtime_error(ss.str());
    }
#endif
  }

  void OpenCLContext::getPlatformDeviceIDsOfType(cl::Platform& platform, 
    const cl_device_type type, std::vector<cl_device_id>& devices) {
    // Unforunately, the ATI C++  API throws a DEEP driver-level memory 
    // exception when querying devices on Nvidia hardware.  Use the C API
    // for this task instead.
    if (devices.size() > 0) {
      devices.clear();
    }

    // First, get the size of device list:
    cl_uint num_devices;
    cl_int rc = clGetDeviceIDs(platform(), type, 0, NULL, &num_devices);
    if (rc != CL_SUCCESS || num_devices == 0) {
      return;
    }

    cl_device_id* dev_ids = new cl_device_id[num_devices];
    CheckError(clGetDeviceIDs(platform(), type, num_devices, dev_ids, NULL));

    for (uint32_t i = 0; i < num_devices; i++) {
      cl_device_type cur_type;
      CheckError(clGetDeviceInfo(dev_ids[i], CL_DEVICE_TYPE, sizeof(cur_type),
        &cur_type, NULL));
      if (cur_type == type) {
        devices.push_back(dev_ids[i]);
      }
    }

    delete[] dev_ids;
  }

  JCLBuffer OpenCLContext::allocateBuffer(const CLBufferType type, 
    const uint32_t nelems) {
    try {
      buffers->pushBack(new OpenCLBufferData(type, nelems, context));
      return (JCLBuffer)(buffers->size() - 1);
    } catch (cl::Error err) {
      throw runtime_error(string("allocateBuffer error: ") + 
        GetCLErrorString(err));
    }
  }

  cl_mem OpenCLContext::getCLMem(const JCLBuffer buffer) {
    if (buffer >= buffers->size()) {
      throw runtime_error("OpenCLContext::getCLMem: Invalid buffer id");
    }
    return (*buffers)[buffer]->buffer()();
  }

  void OpenCLContext::releaseReference(const JCLBuffer buffer) {
    if (buffer >= buffers->size()) {
      throw runtime_error("OpenCLContext::releaseReference: Invalid buffer id");
    }
    (*buffers)[buffer]->releaseReference();
  }

  void OpenCLContext::addReference(const JCLBuffer buffer) {
    if (buffer >= buffers->size()) {
      throw runtime_error("OpenCLContext::addReference: Invalid buffer id");
    }
    (*buffers)[buffer]->addReference();
  }

  void OpenCLContext::useKernel(const char* filename, const char* kernel_name,
    const bool strict_float) {
    // Make sure the program is compiled
    if (cur_program_ == NULL || cur_program_->filename() != filename) {
      if (!programs->lookup(filename, cur_program_)) {
        cur_program_ = new OpenCLProgram(filename, context, devices,
          strict_float); 
        programs->insert(filename, cur_program_);
      }
    }
    // Make sure the Kernel is compiled
    if (cur_kernel_ == NULL || cur_kernel_->program() != cur_program_ ||
      cur_kernel_->kernel_name() != kernel_name) {
      std::string id = std::string(filename) + kernel_name;
      if (!kernels->lookup(id, cur_kernel_)) {
        cur_kernel_ = new OpenCLKernel(kernel_name, cur_program_); 
        kernels->insert(id, cur_kernel_);
      }
    }
  }
  
  void OpenCLContext::useKernelCStr(const char* kernel_c_str,
    const char* kernel_name,  const bool strict_float) {
    // This is a little expensive, but compute the md5 of the kernel_c_str
    // and use this as the "filename"
    unsigned char md5_kernel_c_str[16];
    MD5JCLCStr(kernel_c_str, md5_kernel_c_str);
    std::string filename((char*)md5_kernel_c_str);
    
    // Make sure the program is compiled
    if (cur_program_ == NULL || cur_program_->filename() != filename) {
      if (!programs->lookup(filename, cur_program_)) {
        cur_program_ = new OpenCLProgram(kernel_c_str, filename, context, 
          devices, strict_float);
        programs->insert(filename, cur_program_);
      }
    }
    // Make sure the Kernel is compiled
    if (cur_kernel_ == NULL || cur_kernel_->program() != cur_program_ ||
        cur_kernel_->kernel_name() != kernel_name) {
      std::string id = std::string(filename) + kernel_name;
      if (!kernels->lookup(id, cur_kernel_)) {
        cur_kernel_ = new OpenCLKernel(kernel_name, cur_program_);
        kernels->insert(id, cur_kernel_);
      }
    }
  }

  void OpenCLContext::setArg(const uint32_t index, const JCLBuffer& val) {
    if (!cur_kernel_) {
      throw std::runtime_error("OpenCLContext::setArg() - ERROR: You must "
        "call OpenCL::useKernel() first!");
    }
    cur_kernel_->setArg(index, (*buffers)[(uint32_t)val]->buffer());
  }

  void OpenCLContext::setArg(const uint32_t index, const uint32_t size,
    void* data) {
    if (!cur_kernel_) {
      throw std::runtime_error("OpenCLContext::setArgNull() - ERROR: You must"
        " call OpenCL::useKernel() first!");
    }
    cur_kernel_->setArg(index, size, data);
  }

  void OpenCLContext::sync(const uint32_t device_index) {
#if defined(DEBUG) || defined(_DEBUG)
    if (device_index >= devices.size()) {
      throw std::runtime_error("sync() - ERROR: Invalid device_index");
    }
#endif
    try {
      queues[device_index].finish();
    } catch (cl::Error err) {
      throw runtime_error(string("queues[device_index].finish() failed: ") + 
        GetCLErrorString(err));
    }
  }

  uint32_t OpenCLContext::queryMaxWorkgroupSizeForCurKernel(
    const uint32_t device_index) {
    if (cur_kernel_ == NULL) {
      throw std::runtime_error("queryMaxWorkgroupSizeForCurKernel() - ERROR: "
        "Please call OpenCL::useKernel() first!");
    }
    if (device_index >= devices.size()) {
      throw std::runtime_error("queryMaxWorkgroupSizeForCurKernel() - ERROR: "
        "Invalid device_index");
    }

    size_t max_workgroup_size;
    cl_int rc = cur_kernel_->kernel().getWorkGroupInfo<size_t>(
      devices[device_index], CL_KERNEL_WORK_GROUP_SIZE, &max_workgroup_size);
    if (rc != CL_SUCCESS) {
      throw std::runtime_error("queryMaxWorkgroupSizeForCurKernel() - ERROR: "
        "Failed querying the max workgroup size for this kernel and device!");
    }
    return (uint32_t)max_workgroup_size;
  }

  void OpenCLContext::runKernel(const uint32_t device_index, 
    const uint32_t dim, const uint32_t* global_work_size, 
    const uint32_t* local_work_size, const bool blocking) {
#if defined(DEBUG) || defined(_DEBUG)
    if (device_index >= devices.size()) {
      throw std::runtime_error("runKernel() - ERROR: Invalid "
        "device_index");
    }
    if (cur_kernel_ == NULL) {
      throw std::runtime_error("runKernel() - ERROR: Please call "
        "OpenCL::useKernel() first!");
    }
    if (dim <= 0 || dim > 3) {  // OpenCL doesn't support greater than 3 dims!
      throw std::runtime_error("runKernel() - ERROR: Bad work dims!");
    }
    uint32_t total_worksize = 1;
    for (uint32_t i = 0; i < dim; i++) {
      if ((global_work_size[i] % local_work_size[i]) != 0) {
        throw std::runtime_error("runKernel() - ERROR: Global workgroup"
          " size is not evenly divisible by the local work group size!");
      }
      total_worksize *= local_work_size[i];
      if (local_work_size[i] > (int)devices_max_workitem_size_[device_index][i]) {
        throw std::runtime_error("runKernel() - ERROR: Local workgroup"
          " size is greater than devices_max_workitem_size_!");
      }
    }
    if (total_worksize > (uint32_t)devices_max_workgroup_size_[device_index]) {
      throw std::runtime_error("runKernel() - ERROR: Local workgroup"
        " size is greater than CL_DEVICE_MAX_WORK_GROUP_SIZE!");
    }
    uint32_t max_size = queryMaxWorkgroupSizeForCurKernel(device_index);
    if (total_worksize > (uint32_t)max_size) {
      throw std::runtime_error("runKernel() - ERROR: Local workgroup"
        " size is greater than CL_KERNEL_WORK_GROUP_SIZE!");
    }
#endif

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
    default:
      throw std::runtime_error("runKernel() - ERROR: Bad dim");
    }
    cl::Event cur_event;
    try {
      queues[device_index].enqueueNDRangeKernel(cur_kernel_->kernel(),
        offset, global_work, local_work, NULL, &cur_event);
    } catch (cl::Error err) {
      throw runtime_error(string("enqueueNDRangeKernel failed: ") + 
        GetCLErrorString(err));
    }

    if (blocking) {
      cur_event.wait();
    }
  }

  void OpenCLContext::runKernel(const uint32_t device_index, const uint32_t dim,
    const uint32_t* global_work_size, const bool blocking) {
#if defined(DEBUG) || defined(_DEBUG)
    if (device_index >= devices.size()) {
      throw runtime_error("runKernelxD() - ERROR: Invalid "
        "device_index");
    }
    if (cur_kernel_ == NULL) {
      throw runtime_error("runKernelxD() - ERROR: Please call "
        "OpenCL::useKernel() first!");
    }
    if (dim <= 0 || dim > 3) {  // OpenCL doesn't support greater than 3 dims!
      throw std::runtime_error("runKernel() - ERROR: Bad work dims!");
    }
#endif
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
    default:
      throw std::runtime_error("runKernel() - ERROR: Bad dim");
    }
    cl::Event cur_event;
    try {
      queues[device_index].enqueueNDRangeKernel(cur_kernel_->kernel(),
        offset, global_work, local_work, NULL, &cur_event);
    } catch (cl::Error err) {
      throw runtime_error(string("enqueueNDRangeKernel failed: ") + 
        GetCLErrorString(err));
    }

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
      throw runtime_error("Invalid vendor specified");
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
      throw runtime_error("CLDevice2CLDeviceType() ERROR: Invalid "
        "enumerant!");
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
      throw runtime_error("CLDeviceType2CLDevice() ERROR: Invalid "
        "enumerant!");
    }
    return ret;
  }

  std::string OpenCLContext::GetCLErrorString(const cl::Error& err) {
#if defined(__APPLE__)
    std::cout << "GetCLErrorString() - For more detailed error logging set "
      "the environement var: 'export CL_LOG_ERRORS=stdout'" << std::endl;
#endif
    std::stringstream ss;
    ss << "CLError - type: '" << GetCLErrorEnumString(err.err()) <<
    "'', what: '" << err.what() << "'";
    return ss.str();
  }

  std::string OpenCLContext::GetCLErrorEnumString(const int32_t err) {
    switch (err) {
    case CL_SUCCESS:                          
      return (char*) "Success!";
    case CL_DEVICE_NOT_FOUND:                 
      return (char*) "Device not found.";
    case CL_DEVICE_NOT_AVAILABLE:             
      return (char*) "Device not available";
    case CL_COMPILER_NOT_AVAILABLE:           
      return (char*) "Compiler not available";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:    
      return (char*) "Memory object allocation failure";
    case CL_OUT_OF_RESOURCES:                
      return (char*) "Out of resources";
    case CL_OUT_OF_HOST_MEMORY:               
      return (char*) "Out of host memory";
    case CL_PROFILING_INFO_NOT_AVAILABLE:     
      return (char*) "Profiling information not available";
    case CL_MEM_COPY_OVERLAP:                 
      return (char*) "Memory copy overlap";
    case CL_IMAGE_FORMAT_MISMATCH:            
      return (char*) "Image format mismatch";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:       
      return (char*) "Image format not supported";
    case CL_BUILD_PROGRAM_FAILURE:            
      return (char*) "Program build failure";
    case CL_MAP_FAILURE:                      
      return (char*) "Map failure";
    case CL_INVALID_VALUE:                    
      return (char*) "Invalid value";
    case CL_INVALID_DEVICE_TYPE:              
      return (char*) "Invalid device type";
    case CL_INVALID_PLATFORM:                 
      return (char*) "Invalid platform";
    case CL_INVALID_DEVICE:                   
      return (char*) "Invalid device";
    case CL_INVALID_CONTEXT:                  
      return (char*) "Invalid context";
    case CL_INVALID_QUEUE_PROPERTIES:         
      return (char*) "Invalid queue properties";
    case CL_INVALID_COMMAND_QUEUE:            
      return (char*) "Invalid command queue";
    case CL_INVALID_HOST_PTR:                 
      return (char*) "Invalid host pointer";
    case CL_INVALID_MEM_OBJECT:               
      return (char*) "Invalid memory object";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  
      return (char*) "Invalid image format descriptor";
    case CL_INVALID_IMAGE_SIZE:               
      return (char*) "Invalid image size";
    case CL_INVALID_SAMPLER:                  
      return (char*) "Invalid sampler";
    case CL_INVALID_BINARY:                   
      return (char*) "Invalid binary";
    case CL_INVALID_BUILD_OPTIONS:            
      return (char*) "Invalid build options";
    case CL_INVALID_PROGRAM:                  
      return (char*) "Invalid program";
    case CL_INVALID_PROGRAM_EXECUTABLE:       
      return (char*) "Invalid program executable";
    case CL_INVALID_KERNEL_NAME:              
      return (char*) "Invalid kernel name";
    case CL_INVALID_KERNEL_DEFINITION:        
      return (char*) "Invalid kernel definition";
    case CL_INVALID_KERNEL:                   
      return (char*) "Invalid kernel";
    case CL_INVALID_ARG_INDEX:                
      return (char*) "Invalid argument index";
    case CL_INVALID_ARG_VALUE:                
      return (char*) "Invalid argument value";
    case CL_INVALID_ARG_SIZE:                 
      return (char*) "Invalid argument size";
    case CL_INVALID_KERNEL_ARGS:              
      return (char*) "Invalid kernel arguments";
    case CL_INVALID_WORK_DIMENSION:           
      return (char*) "Invalid work dimension";
    case CL_INVALID_WORK_GROUP_SIZE:          
      return (char*) "Invalid work group size";
    case CL_INVALID_WORK_ITEM_SIZE:           
      return (char*) "Invalid work item size";
    case CL_INVALID_GLOBAL_OFFSET:           
      return (char*) "Invalid global offset";
    case CL_INVALID_EVENT_WAIT_LIST:          
      return (char*) "Invalid event wait list";
    case CL_INVALID_EVENT:                    
      return (char*) "Invalid event";
    case CL_INVALID_OPERATION:                
      return (char*) "Invalid operation";
    case CL_INVALID_GL_OBJECT:                
      return (char*) "Invalid OpenGL object";
    case CL_INVALID_BUFFER_SIZE:              
      return (char*) "Invalid buffer size";
    case CL_INVALID_MIP_LEVEL:                
      return (char*) "Invalid mip-map level";
    default:                                  
      return (char*) "Unknown";
    }
  }

}  // namespace jcl
