//
//  tensor.h
//
//  Created by Jonathan Tompson on 5/14/13.
//
//  Simplified C++ replica of torch.Tensor.  Up to 3D is supported.
//
//  This is escentially a wrap around my opencl buffer class.
//

#pragma once


#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "jcl/math/int_types.h"
#include "jcl/math/math_types.h"
#include "jcl/jcl.h"  // For jcl::JCLBuffer
#include "jtorch/torch_data.h"
#include "jtorch/jtorch.h"

#define JTORCH_TENSOR_PRECISON 4

namespace jcl {
namespace threading {
class ThreadPool;
}
}

#define TO_TENSOR_PTR(x)                                                 \
  ((x != nullptr && ((x)->type() == jtorch::TorchDataType::TENSOR_DATA)) \
       ? (jtorch::Tensor<float>*)(x)                                     \
       : nullptr)

namespace jtorch {

static const char* kFillKernel =
"    __kernel void Fill("
"      __global float* output,  /* 0 */"
"      const float value) {     /* 1 */"
"      const int x_out = get_global_id(0);"
"      output[x_out] = value;"
"    }";

static const char* kDivKernel =
"    /* output = output / div_val */"
"    __kernel void Div("
"      const  float div_val,  /* 0 */"
"      __global  float* output) {      /* 1 */"
"      const int x_out = get_global_id(0);"
"      output[x_out] /= div_val;"
"    }";

static const char* kAccumulateKernel =
"    /* output += input1 */"
"    __kernel void Accumulate("
"      const __global  float* input1,  /* 0 */"
"      __global  float* output) {      /* 2 */"
"      const int x_out = get_global_id(0);"
"      output[x_out] += input1[x_out];"
"    }";

static const char* kAddKernel =
"    /* output = input1 + input2 */"
"    __kernel void Add("
"      const __global  float* input1,  /* 0 */"
"      const __global  float* input2,  /* 1 */"
"      __global  float* output) {      /* 2 */"
"      const int x_out = get_global_id(0);"
"      output[x_out] = input1[x_out] + input2[x_out];"
"    }";

static const char* kSubKernel =
"    /* output = input1 - input2 */"
"    __kernel void Sub("
"      const __global  float* input1,  /* 0 */"
"      const __global  float* input2,  /* 1 */"
"      __global  float* output) {      /* 2 */"
"      const int x_out = get_global_id(0);"
"      output[x_out] = input1[x_out] - input2[x_out];"
"    }";

static const char* kAbsKernel =
"    /* output = |input1| */"
"    __kernel void Abs("
"      const __global  float* input1,  /* 0 */"
"      __global  float* output) {     /* 1 */"
"      const int x_out = get_global_id(0);"
"      output[x_out] = fabs(input1[x_out]);"
"    }";

static const char* kCopyKernel =
"    __kernel void Copy("
"      const __global float* input,  /* 0 */"
"      __global float* output) {     /* 1 */"
"      const int x_out = get_global_id(0);"
"      output[x_out] = input[x_out];"
"    }";

static const char* kMulKernel =
"    /* output = mul_val * output */"
"    __kernel void Mul("
"      const  float mul_val,  /* 0 */"
"      __global  float* output) {      /* 1 */"
"      const int x_out = get_global_id(0);"
"      output[x_out] *= mul_val;"
"    }";

template <typename T>
class Tensor : public TorchData {
 public:
  Tensor(const uint32_t dim, const uint32_t* size);
  ~Tensor() override;

  TorchDataType type() const override { return TENSOR_DATA; }

  // setData and getData are EXPENSIVE --> They require a CPU to GPU copy
  void setData(const T* data);
  void getData(T* data) const;

  // View returns a new view on the same object.  The caller owns the new
  // memory (ie, it is transferred).
  std::shared_ptr<Tensor<T>> view(const uint32_t dim, const uint32_t* size);

  const uint32_t dim() const { return dim_; }
  const uint32_t* size() const { return size_.get(); }
  const bool isSameSizeAs(const Tensor<T>& src) const;

  // Print --> EXPENSIVE
  void print() override;  // print to std::cout

  // Some simple tensor math operations
  // copy: dst = src
  static void copy(Tensor<T>& dst, const Tensor<T>& src);
  // add: dst = x + y
  static void add(Tensor<T>& dst, const Tensor<T>& x, const Tensor<T>& y);
  // sub: dst = x - y
  static void sub(Tensor<T>& dst, const Tensor<T>& x, const Tensor<T>& y);
  // abs: x = |x|
  static void abs(Tensor<T>& x);
  // mul: x = x * mul_value
  static void mul(Tensor<T>& x, float mul_value);
  // div: x = x / div_value
  static void div(Tensor<T>& x, float div_value);
  // accumulate: dst += src
  static void accumulate(Tensor<T>& dst, const Tensor<T>& src);
  // zero: x = vec(0)
  static void zero(Tensor<T>& x);
  // fill: x = vec(value)
  static void fill(Tensor<T>& x, float value);
  // slowSum - This does a CPU copy because I haven't written a reduction
  // operator yet
  static float slowSum(const Tensor<T>& x);
  // slowMax - This does a CPU copy because I haven't written a reduction
  // operator yet
  static float slowMax(const Tensor<T>& x);
  // slowMin - This does a CPU copy because I haven't written a reduction
  // operator yet
  static float slowMin(const Tensor<T>& x);

  // Some tensor math operations that return new tensors
  static Tensor<T>* clone(const Tensor<T>& x);
  static Tensor<T>* gaussian1D(const int32_t kernel_size);  // sigma = size / 2
  static Tensor<T>* gaussian(const int32_t kernel_size);
  static Tensor<T>* loadFromFile(const std::string& file);
  static void saveToFile(const Tensor<T>* tensor, const std::string& file);

  inline const jcl::JCLBuffer& storage() const { return storage_; }
  inline uint32_t nelems() const;

  uint32_t* calcStride() const;  // memory returned is owned by caller

 protected:
  jcl::JCLBuffer storage_;  // Internal data
  uint32_t dim_;
  std::unique_ptr<uint32_t[]> size_;  // size_[0] is lowest contiguous dimension,
                                      // size_[2] is highest dimension

  Tensor();  // Default constructor used internally (in view function)

  // Non-copyable, non-assignable.
  Tensor(Tensor&);
  Tensor& operator=(const Tensor&);
};

template <typename T>
Tensor<T>::Tensor(const uint32_t dim, const uint32_t* size) {
  this->dim_ = dim;
  this->size_.reset(new uint32_t[dim]);
  memcpy(this->size_.get(), size, sizeof(this->size_[0]) * dim);
  storage_ = jtorch::cl_context->allocateBuffer(
      jcl::CLBufferTypeReadWrite,
      nelems());  // Adds a reference to the reference count
  zero(*this);
}

template <typename T>
Tensor<T>::Tensor() {
  // Default constructor returns an empty header.  Used internally (ie
  // private).
  dim_ = 0;
  size_.reset(nullptr);
  storage_ = (jcl::JCLBuffer)-1;
}

template <typename T>
Tensor<T>::~Tensor() {
  // Note: If the following assertions are breaking, it means
  // that you are not cleaning up your allocated tensors
  // before shutting down jtorch.
  RASSERT(jtorch::cl_context != nullptr);
  jtorch::cl_context->releaseReference(storage_);
}

template <typename T>
uint32_t* Tensor<T>::calcStride() const {
  uint32_t* stride = new uint32_t[dim_];
  stride[0] = 1;
  for (uint32_t i = 1; i < dim_; i++) {
    stride[i] = stride[i - 1] * size_[i - 1];
  }
  return stride;
}

template <typename T>
uint32_t Tensor<T>::nelems() const {
  uint32_t nelem = 1;
  for (uint32_t i = 0; i < dim_; i++) {
    nelem *= size_[i];
  }
  return nelem;
}

template <typename T>
const bool Tensor<T>::isSameSizeAs(const Tensor<T>& src) const {
  if (dim_ != src.dim_) {
    return false;
  }
  for (uint32_t i = 0; i < dim_; i++) {
    if (size_[i] != src.size_[i]) {
      return false;
    }
  }
  return true;
}

template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::view(const uint32_t dim,
                                           const uint32_t* size) {
  RASSERT(dim != 0);
  int32_t view_nelem = 1;
  for (uint32_t i = 0; i < dim; i++) {
    view_nelem *= size[i];
  }

  RASSERT(view_nelem == nelems());  // Otherwise size mismatch

  std::shared_ptr<Tensor<T>> return_header(new Tensor<T>());
  return_header->dim_ = dim;
  return_header->size_.reset(new uint32_t[dim]);
  memcpy(return_header->size_.get(), size, sizeof(return_header->size_[0]) * dim);
  return_header->storage_ = storage_;
  jtorch::cl_context->addReference(storage_);
  return return_header;
}

template <typename T>
void Tensor<T>::setData(const T* data) {
  jtorch::cl_context->writeToBuffer(data, jtorch::deviceid, storage_, true);
}

template <typename T>
void Tensor<T>::getData(T* data) const {
  jtorch::cl_context->readFromBuffer(data, jtorch::deviceid, storage_, true);
}

template <typename T>
void Tensor<T>::print() {
  std::streamsize prec = std::cout.precision();
  std::cout.precision(JTORCH_TENSOR_PRECISON);
  T* d = new T[nelems()];
  getData(d);
  T max_val = std::numeric_limits<T>::min();
  for (uint32_t i = 0; i < nelems(); i++) {
    max_val = std::max<T>(max_val, d[i]);
  }
  T scale = (T)pow(10.0, floor(log10((double)max_val + kEpsilon)));

#if defined(WIN32) || defined(_WIN32)
  std::cout.setf(0, std::ios::showpos);
#else
  std::cout.setf(std::ios::showpos);
#endif

  if (dim_ == 1) {
    // Print a 1D tensor
    std::cout << "  tensor[*] =" << std::endl;
    if (fabsf((float)scale - 1.0f) > kEpsilon) {
      std::cout << " " << scale << " * " << std::endl;
    }
    std::cout.setf(std::ios::showpos);
    for (uint32_t u = 0; u < size_[0]; u++) {
      if (u == 0) {
        std::cout << " (0) ";
      } else {
        std::cout << "     ";
      }
      std::cout << std::fixed << d[u] / scale << std::endl;
      ;
      std::cout.unsetf(std::ios_base::floatfield);
    }
  } else if (dim_ == 2) {
    // Print a 2D tensor
    std::cout << "  tensor[*,*] =" << std::endl;
    if (fabsf((float)scale - 1.0f) > kEpsilon) {
      std::cout << " " << scale << " * " << std::endl;
    }
    std::cout.setf(std::ios::showpos);
    for (uint32_t v = 0; v < size_[1]; v++) {
      if (v == 0) {
        std::cout << " (0,0) ";
      } else {
        std::cout << "       ";
      }
      std::cout.setf(std::ios::showpos);
      for (uint32_t u = 0; u < size_[0]; u++) {
        std::cout << std::fixed << d[v * size_[0] + u] / scale;
        std::cout.unsetf(std::ios_base::floatfield);
        if (u != size_[0] - 1) {
          std::cout << ", ";
        } else {
          std::cout << std::endl;
        }
      }
    }
  } else {
    // Print a nD tensor
    int32_t odim = 1;
    for (uint32_t i = 2; i < dim_; i++) {
      odim *= size_[i];
    }

    uint32_t* stride = calcStride();

    for (int32_t i = 0; i < odim; i++) {
      std::cout << "  tensor[";
      for (uint32_t cur_dim = dim_ - 1; cur_dim >= 2; cur_dim--) {
        std::cout << (i % stride[cur_dim]) << ",";
      }

      std::cout << "*,*] =";
      std::cout << std::endl;
      if (fabsf((float)scale - 1.0f) > kEpsilon) {
        std::cout << " " << scale << " * " << std::endl;
      }

      T* data = &d[i * size_[1] * size_[0]];
      for (uint32_t v = 0; v < size_[1]; v++) {
        if (v == 0) {
          std::cout << " (0,0) ";
        } else {
          std::cout << "       ";
        }
        std::cout.setf(std::ios::showpos);
        for (uint32_t u = 0; u < size_[0]; u++) {
          std::cout << std::fixed << data[v * size_[0] + u] / scale;
          std::cout.unsetf(std::ios_base::floatfield);
          if (u != size_[0] - 1) {
            std::cout << ", ";
          } else {
            std::cout << std::endl;
          }
        }
      }
    }

    delete[] stride;
  }
  std::cout.precision(prec);
  std::cout << std::resetiosflags(std::ios_base::showpos);
  delete[] d;

  std::cout << "[jtorch.";
  std::cout << jcl::JCL::CLDeviceToString(jtorch::cl_context->device());
  std::cout << " of dimension ";
  for (int32_t i = (int32_t)dim_ - 1; i >= 0; i--) {
    std::cout << size_[i];
    if (i > 0) {
      std::cout << "x";
    }
  }
  std::cout << "]" << std::endl;
};

template <typename T>
Tensor<T>* Tensor<T>::gaussian1D(const int32_t kernel_size) {
  const uint32_t size = kernel_size;
  Tensor<T>* ret = new Tensor<T>(1, &size);
  const float sigma = 0.25f;
  const float amplitude = 1.0f;
  const float center = (float)kernel_size / 2.0f + 0.5f;
  T* data = new T[kernel_size];
  for (int32_t i = 0; i < kernel_size; i++) {
    data[i] =
        (T)amplitude *
        expf(-(powf(((float)(i + 1) - center) / (sigma * (float)kernel_size),
                    2.0f) /
               2.0f));
  }
  ret->setData(data);
  delete[] data;
  return ret;
}

template <typename T>
Tensor<T>* Tensor<T>::gaussian(const int32_t kernel_size) {
  const uint32_t size[2] = {kernel_size, kernel_size};
  Tensor<T>* ret = new Tensor<T>(2, size);
  const float sigma = 0.25f;
  const float amplitude = 1.0f;
  const float center = (float)kernel_size / 2.0f + 0.5f;
  T* data = new T[kernel_size * kernel_size];
  for (int32_t v = 0; v < kernel_size; v++) {
    for (int32_t u = 0; u < kernel_size; u++) {
      float du = ((float)(u + 1) - center) / (sigma * (float)kernel_size);
      float dv = ((float)(v + 1) - center) / (sigma * (float)kernel_size);
      data[v * kernel_size + u] =
          (T)amplitude * expf(-(du * du + dv * dv) / 2.0f);
    }
  }
  ret->setData(data);
  delete[] data;
  return ret;
}

template <typename T>
Tensor<T>* Tensor<T>::clone(const Tensor<T>& x) {
  Tensor<T>* ret = new Tensor<T>(x.dim_, x.size_.get());
  cl_context->useKernelCStr(kCopyKernel, "Copy");
  cl_context->setArg(0, x.storage());
  cl_context->setArg(1, ret->storage());
  uint32_t dim = 1;
  uint32_t nelem = x.nelems();
  cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);
  return ret;
}

template <typename T>
void Tensor<T>::copy(Tensor<T>& dst, const Tensor<T>& src) {
  cl_context->useKernelCStr(kCopyKernel, "Copy");
  cl_context->setArg(0, src.storage());
  cl_context->setArg(1, dst.storage());
  uint32_t dim = 1;
  uint32_t nelem = dst.nelems();
  cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);
}

template <typename T>
void Tensor<T>::add(Tensor<T>& dst, const Tensor<T>& x, const Tensor<T>& y) {
  cl_context->useKernelCStr(kAddKernel, "Add");
  cl_context->setArg(0, x.storage());
  cl_context->setArg(1, y.storage());
  cl_context->setArg(2, dst.storage());
  uint32_t dim = 1;
  uint32_t nelem = dst.nelems();
  cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);
}

template <typename T>
void Tensor<T>::sub(Tensor<T>& dst, const Tensor<T>& x, const Tensor<T>& y) {
  cl_context->useKernelCStr(kSubKernel, "Sub");
  cl_context->setArg(0, x.storage());
  cl_context->setArg(1, y.storage());
  cl_context->setArg(2, dst.storage());
  uint32_t dim = 1;
  uint32_t nelem = dst.nelems();
  cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);
}

template <typename T>
void Tensor<T>::abs(Tensor<T>& x) {
  cl_context->useKernelCStr(kAbsKernel, "Abs");
  cl_context->setArg(0, x.storage());
  uint32_t dim = 1;
  uint32_t nelem = x.nelems();
  cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);
}

template <typename T>
void Tensor<T>::mul(Tensor<T>& x, float mul_val) {
  cl_context->useKernelCStr(kMulKernel, "Mul");
  cl_context->setArg(0, mul_val);
  cl_context->setArg(1, x.storage());
  uint32_t dim = 1;
  uint32_t nelem = x.nelems();
  cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);
}

template <typename T>
void Tensor<T>::div(Tensor<T>& x, float div_val) {
  cl_context->useKernelCStr(kDivKernel, "Div");
  cl_context->setArg(0, div_val);
  cl_context->setArg(1, x.storage());
  uint32_t dim = 1;
  uint32_t nelem = x.nelems();
  cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);
}

template <typename T>
void Tensor<T>::accumulate(Tensor<T>& dst, const Tensor<T>& src) {
  cl_context->useKernelCStr(kAccumulateKernel, "Accumulate");
  cl_context->setArg(0, src.storage());
  cl_context->setArg(1, dst.storage());
  uint32_t dim = 1;
  uint32_t nelem = dst.nelems();
  cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);
}

template <typename T>
void Tensor<T>::zero(Tensor<T>& dst) {
  Tensor<T>::fill(dst, 0);
}

template <typename T>
void Tensor<T>::fill(Tensor<T>& dst, float value) {
  cl_context->useKernelCStr(kFillKernel, "Fill");
  cl_context->setArg(0, dst.storage());
  cl_context->setArg(1, value);
  uint32_t dim = 1;
  uint32_t nelem = dst.nelems();
  cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);
}

template <typename T>
float Tensor<T>::slowSum(const Tensor<T>& x) {
  std::unique_ptr<float[]> temp(new float[x.nelems()]);
  x.getData(temp.get());
  float sum = 0.0f;
  for (uint32_t i = 0; i < x.nelems(); i++) {
    sum += temp[i];
  }
  return sum;
}

template <typename T>
float Tensor<T>::slowMax(const Tensor<T>& x) {
  std::unique_ptr<float[]> temp(new float[x.nelems()]);
  x.getData(temp.get());
  float max = -std::numeric_limits<float>::infinity();
  for (uint32_t i = 0; i < x.nelems(); i++) {
    if (max < temp[i]) {
      max = temp[i];
    }
  }
  return max;
}

template <typename T>
float Tensor<T>::slowMin(const Tensor<T>& x) {
  std::unique_ptr<float[]> temp(new float[x.nelems()]);
  x.getData(temp.get());
  float min = std::numeric_limits<float>::infinity();
  for (uint32_t i = 0; i < x.nelems(); i++) {
    if (min > temp[i]) {
      min = temp[i];
    }
  }
  return min;
}

template <typename T>
Tensor<T>* Tensor<T>::loadFromFile(const std::string& file) {
  Tensor<T>* new_tensor = nullptr;
  std::ifstream ifile(file.c_str(), std::ios::in | std::ios::binary);
  if (ifile.is_open()) {
    ifile.seekg(0, std::ios::beg);
    // Now load the Tensor
    int32_t dim;
    ifile.read((char*)(&dim), sizeof(dim));
    uint32_t* size = new uint32_t[dim];
    for (int32_t i = 0; i < dim; i++) {
      int32_t cur_size;
      ifile.read((char*)(&cur_size), sizeof(cur_size));
      size[dim - i - 1] = (uint32_t)cur_size;
    }
    new_tensor = new Tensor<T>(dim, size);

    T* data = new T[new_tensor->nelems()];
    ifile.read((char*)(data), sizeof(data[0]) * new_tensor->nelems());
    new_tensor->setData(data);
    delete[] data;
    ifile.close();
    delete[] size;
  } else {
    std::cout << "Tensor<T>::loadFromFile() - ERROR: Could not open file ";
    std::cout << file << std::endl;
    RASSERT(false);
    return nullptr;
  }
  return new_tensor;
}

template <typename T>
void Tensor<T>::saveToFile(const Tensor<T>* tensor, const std::string& file) {
  std::ofstream ofile(file.c_str(), std::ios::out | std::ios::binary);
  if (ofile.is_open()) {
    // Now save the Tensor
    int32_t dim = tensor->dim_;
    ofile.write((char*)(&dim), sizeof(dim));
    for (int32_t i = dim - 1; i >= 0; i--) {
      int32_t cur_size = tensor->size_[i];
      ofile.write((char*)(&cur_size), sizeof(cur_size));
    }
    T* data = new T[tensor->nelems()];
    tensor->getData(data);
    ofile.write((char*)(data), sizeof(data[0]) * tensor->nelems());
    delete[] data;
    ofile.close();
  } else {
    std::cout << "Tensor<T>::saveToFile() - ERROR: Could not open file ";
    std::cout << file << std::endl;
    RASSERT(false);
  }
}

};  // namespace jtorch
