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
#include <limits>
#include <random>
#include <sstream>

#include "jcl/math/int_types.h"
#include "jcl/math/math_types.h"
#include "jcl/opencl_buffer_data.h"
#include "jcl/opencl_context.h"
#include "jtorch/jtorch.h"
#include "jtorch/torch_data.h"

#define JTORCH_TENSOR_PRECISON 4

namespace jcl {
namespace threading {
class ThreadPool;
}  // namespace threading
}  // namespace jcl

#define TO_TENSOR_PTR(x)                                                 \
  ((x != nullptr && ((x)->type() == jtorch::TorchDataType::TENSOR_DATA)) \
       ? (jtorch::Tensor<float>*)(x)                                     \
       : nullptr)

namespace jtorch {

static const char* kFillKernel =
"    __kernel void Fill(\n"
"      __global float* output,  /* 0 */\n"
"      const float value) {     /* 1 */\n"
"      const int x_out = get_global_id(0);\n"
"      output[x_out] = value;\n"
"    }";

static const char* kAccumulateKernel =
"    /* output += input1 */\n"
"    __kernel void Accumulate(\n"
"      const __global  float* input1,  /* 0 */\n"
"      __global  float* output) {      /* 2 */\n"
"      const int x_out = get_global_id(0);\n"
"      output[x_out] += input1[x_out];\n"
"    }";

static const char* kAddKernel =
"    /* output = input1 + input2 */\n"
"    __kernel void Add(\n"
"      const __global  float* input1,  /* 0 */\n"
"      const __global  float* input2,  /* 1 */\n"
"      __global  float* output) {      /* 2 */\n"
"      const int x_out = get_global_id(0);\n"
"      output[x_out] = input1[x_out] + input2[x_out];\n"
"    }";

static const char* kSubKernel =
"    /* output = input1 - input2 */\n"
"    __kernel void Sub(\n"
"      const __global  float* input1,  /* 0 */\n"
"      const __global  float* input2,  /* 1 */\n"
"      __global  float* output) {      /* 2 */\n"
"      const int x_out = get_global_id(0);\n"
"      output[x_out] = input1[x_out] - input2[x_out];\n"
"    }";

static const char* kAbsKernel =
"    /* output = |input1| */\n"
"    __kernel void Abs(\n"
"      __global  float* output) {     /* 0 */\n"
"      const int x_out = get_global_id(0);\n"
"      output[x_out] = fabs(output[x_out]);\n"
"    }";

static const char* kCopyKernel =
"    __kernel void Copy(\n"
"      const __global float* input,  /* 0 */\n"
"      __global float* output) {     /* 1 */\n"
"      const int x_out = get_global_id(0);\n"
"      output[x_out] = input[x_out];\n"
"    }";

static const char* kMulKernel =
"    /* output = mul_val * output */\n"
"    __kernel void Mul(\n"
"      const  float mul_val,  /* 0 */\n"
"      __global  float* output) {      /* 1 */\n"
"      const int x_out = get_global_id(0);\n"
"      output[x_out] *= mul_val;\n"
"    }";

static const char* kAddScalarKernel =
"    /* output = add_val + output */\n"
"    __kernel void AddScalarKernel(\n"
"      const  float add_val,  /* 0 */\n"
"      __global  float* output) {      /* 1 */\n"
"      const int x_out = get_global_id(0);\n"
"      output[x_out] += add_val;\n"
"    }";

// Note: This Tensor class DOESN'T support non-contiguous tensors.  Updating
// it to do so wouldn't be a huge amount of work, but I have not needed to
// do any select or narrow operations on the inner dimensions, so I have
// not implemented it.
// TODO(tompson): Implement non-contiguous support!
template <typename T>
class Tensor : public TorchData {
 public:
  // Default constructor that allocates a zero dimension tensor.
  Tensor();
  // Constructor to allocate a tensor of dimension dim. size is an array of
  // sizes for each dimension. size[0] is the lowest (contiguous) dimension.
  // Note that this is opposite to torch, where size(1) is the highest (outer)
  // dimension.
  Tensor(const uint32_t dim, const uint32_t* size);
  ~Tensor() override;

  TorchDataType type() const override { return TENSOR_DATA; }

  // setData and getData are EXPENSIVE --> They require a CPU to GPU copy
  void setData(const T* data);
  void getData(T* data) const;

  const uint32_t dim() const { return dim_; }
  const uint32_t* size() const { return size_.get(); }
  const bool isSameSizeAs(const Tensor<T>& src) const;

  // As per torch convention, resize does not allocate a new buffer if the
  // requested size is smaller than (or equal to) the current size. Otherwise it
  // will allocate a new buffer and copy over the elements. True is returned
  // when the underlining storage changes (i.e. memory was allocated).
  bool resize(const uint32_t dim, const uint32_t* size);

  // Courtesy function for the above.
  bool resizeAs(const Tensor<T>& src);

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
  // add: dst = x + add_value
  static void add(Tensor<T>& x, float add_value);
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
  // slowMean - This does a CPU copy because I haven't written a reduction
  // operator yet
  static float slowMean(const Tensor<T>& x);
  // slowRand - This generates random numbers on the CPU then uploads them.
  static std::shared_ptr<Tensor<T>> slowRand(const uint32_t dim,
                                             const uint32_t* size);

  // Some tensor math operations that return new tensors
  static std::shared_ptr<Tensor<T>> clone(const Tensor<T>& x);
  // for gaussian1D: sigma = size / 2
  static std::shared_ptr<Tensor<T>> gaussian1D(const int32_t kernel_size);
  static std::shared_ptr<Tensor<T>> gaussian(const int32_t kernel_size);
  static std::shared_ptr<Tensor<T>> loadFromFile(const std::string& file);
  static void saveToFile(const Tensor<T>& tensor, const std::string& file);

  // In selectOuterDim we do not fully support slicing tensors along any
  // arbitrary dimension, but it is easy enough to select contiguous chunks.
  // As per torch standard, select reduces dimension by 1.
  static std::shared_ptr<Tensor<T>> selectOuterDim(const Tensor<T>& src,
                                                   const uint32_t i);

  // In narrowOuterDim we do not fully support slicing tensors along any
  // arbitrary dimension, but it is easy enough to select contiguous chunks.
  // As per torch standard, narrow does not reduce dimension by 1.
  static std::shared_ptr<Tensor<T>> narrowOuterDim(const Tensor<T>& src,
                                                   const uint32_t i,
                                                   const uint32_t length);

  // View returns a new view on the same object.  The caller owns the new
  // memory (ie, it is transferred).
  static std::shared_ptr<Tensor<T>> view(const Tensor<T>& src,
                                         const uint32_t dim,
                                         const uint32_t* size);

  const std::shared_ptr<jcl::OpenCLBufferData> storage() const;
  inline uint32_t nelems() const;
  std::unique_ptr<uint32_t[]> calcStride() const;

 protected:
  std::shared_ptr<jcl::OpenCLBufferData> storage_;  // Internal data
  uint32_t dim_;
  std::unique_ptr<uint32_t[]> size_;  // size_[0] is lowest contiguous dim,
                                      // size_[2] is highest dim

  // Non-copyable, non-assignable.
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
};

template <typename T>
Tensor<T>::Tensor(const uint32_t dim, const uint32_t* size) {
  this->dim_ = dim;
  this->size_.reset(new uint32_t[dim]);
  memcpy(this->size_.get(), size, sizeof(this->size_[0]) * dim);
  storage_ = jtorch::cl_context->allocateBuffer(
      jcl::CLBufferTypeReadWrite,
      nelems());
  zero(*this);
}

template <typename T>
Tensor<T>::Tensor() {
  // Default constructor returns an empty header.  Used internally (ie
  // private).
  dim_ = 0;
  size_.reset(nullptr);
  storage_ = nullptr;
}

template <typename T>
Tensor<T>::~Tensor() {
  // Note: If the following assertions are breaking, it means
  // that you are not cleaning up your allocated tensors
  // before shutting down jtorch.
  RASSERT(jtorch::cl_context != nullptr);
  storage_ = nullptr;  // decrement ref count
}

template <typename T>
const std::shared_ptr<jcl::OpenCLBufferData> Tensor<T>::storage() const {
  return storage_;
}

template <typename T>
bool Tensor<T>::resize(const uint32_t dim, const uint32_t* size) {
  RASSERT(dim > 0);
  uint32_t new_nelems = size[0];
  for (uint32_t i = 1; i < dim; i++) {
    new_nelems *= size[i];
  }
  bool new_alloc = false;
  if (storage_ == nullptr || storage_->nelems() < new_nelems) {
    // The user requested a larger tensor. We need to allocate a larger tensor
    // and copy over what we have.
    std::shared_ptr<jcl::OpenCLBufferData> new_storage =
        jtorch::cl_context->allocateBuffer(jcl::CLBufferTypeReadWrite,
                                           new_nelems);
    if (storage_ != nullptr) {
      cl_context->useKernelCStr(kCopyKernel, "Copy");
      cl_context->setArg(0, storage_);     // input
      cl_context->setArg(1, new_storage);  // ouptut
      uint32_t dim = 1;
      // The current view might be smaller than the old storage, so avoid
      // copying too much data.
      uint32_t nelem = nelems();
      cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);
    }
    storage_ = new_storage;
    new_alloc = true;
  } else {
    // Otherwise, the size is smaller than the current storage, so just shrink
    // the view.
  }
  if (dim_ != dim) {
    size_.reset(new uint32_t[dim]);
  }
  dim_ = dim;
  for (uint32_t i = 0; i < dim_; i++) {
    size_[i] = size[i];
  }
  return new_alloc;
}

template <typename T>
bool Tensor<T>::resizeAs(const Tensor<T>& src) {
  return resize(src.dim_, src.size_.get());
}

template <typename T>
std::unique_ptr<uint32_t[]> Tensor<T>::calcStride() const {
  std::unique_ptr<uint32_t[]> stride(new uint32_t[dim_]);
  stride[0] = 1;
  for (uint32_t i = 1; i < dim_; i++) {
    stride[i] = stride[i - 1] * size_[i - 1];
  }
  return std::move(stride);
}

template <typename T>
uint32_t Tensor<T>::nelems() const {
  if (dim_ == 0) {
    return 0;
  }
  uint32_t nelem = 1;
  for (uint32_t i = 0; i < dim_; i++) {
    nelem *= size_[i];
  }
  return nelem;
}

template <typename T>
const bool Tensor<T>::isSameSizeAs(const Tensor<T>& src) const {
  if (dim_ != src.dim_) {
    std::cout << "here" << std::endl;
    return false;
  }
  for (uint32_t i = 0; i < dim_; i++) {
    if (size_[i] != src.size_[i]) {
      std::cout << "here" << std::endl;
      std::cout << size_[i] << std::endl;
      std::cout << src.size_[i] << std::endl;
      return false;
    }
  }
  return true;
}

template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::view(const Tensor<T>& src,
                                           const uint32_t dim,
                                           const uint32_t* size) {
  RASSERT(dim != 0);
  int32_t view_nelem = 1;
  for (uint32_t i = 0; i < dim; i++) {
    view_nelem *= size[i];
  }

  RASSERT(view_nelem == src.nelems());  // Otherwise size mismatch

  std::shared_ptr<Tensor<T>> return_header(new Tensor<T>());
  return_header->dim_ = dim;
  return_header->size_.reset(new uint32_t[dim]);
  memcpy(return_header->size_.get(), size,
         sizeof(return_header->size_[0]) * dim);
  return_header->storage_ = src.storage_;  // Incruments ref count.
  return return_header;
}

template <typename T>
void Tensor<T>::setData(const T* data) {
  RASSERT(dim_ != 0);
  jtorch::cl_context->writeToBuffer(data, nelems(), jtorch::deviceid, storage_,
                                    true);
}

template <typename T>
void Tensor<T>::getData(T* data) const {
  RASSERT(dim_ != 0);
  jtorch::cl_context->readFromBuffer(data, nelems(), jtorch::deviceid, storage_,
                                     true);
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

  if (dim_ == 0) {
    std::cout << "  tensor[]" << std::endl;
  } else if (dim_ == 1) {
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
    RASSERT(dim_ > 2);
    // Print a nD tensor
    int32_t odim = 1;  // Number of outer dimensions greater than 2.
    for (uint32_t i = 2; i < dim_; i++) {
      odim *= size_[i];
    }

    const uint32_t upper_dim = dim_ - 2;
    std::unique_ptr<uint32_t[]> upper_stride(new uint32_t[upper_dim]);
    std::unique_ptr<uint32_t[]> upper_size(new uint32_t[upper_dim]);
    upper_stride[0] = 1;
    upper_size[0] = size_[2];
    for (uint32_t i = 1; i < upper_dim; i++) {
      upper_size[i] = size_[i + 2];
      upper_stride[i] = upper_stride[i - 1] * upper_size[i - 1];
    }

    for (int32_t i = 0; i < odim; i++) {
      std::cout << "  tensor[";
      for (int32_t cur_dim = upper_dim - 1; cur_dim >= 0; cur_dim--) {
        if (upper_size[cur_dim] == 1) {
          std::cout << "1,";
        } else {
          std::cout << ((i / upper_stride[cur_dim]) % upper_size[cur_dim])
                    << ",";
        }
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
  }
  std::cout.precision(prec);
  std::cout << std::resetiosflags(std::ios_base::showpos);
  delete[] d;

  std::cout << "[jtorch.";
  std::cout << jcl::OpenCLContext::CLDeviceToString(
      jtorch::cl_context->getDeviceType(jtorch::deviceid));
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
std::shared_ptr<Tensor<T>> Tensor<T>::gaussian1D(const int32_t kernel_size) {
  const uint32_t size = kernel_size;
  std::shared_ptr<Tensor<T>> ret(new Tensor<T>(1, &size));
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
  return std::shared_ptr<Tensor<T>>(ret);
}

template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::gaussian(const int32_t kernel_size) {
  const uint32_t size[2] = {(uint32_t)kernel_size, (uint32_t)kernel_size};
  std::shared_ptr<Tensor<T>> ret(new Tensor<T>(2, size));
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
  std::shared_ptr<Tensor<T>> Tensor<T>::clone(const Tensor<T>& x) {
  RASSERT(x.dim_ != 0);
  std::shared_ptr<Tensor<T>> ret(new Tensor<T>(x.dim_, x.size_.get()));
  cl_context->useKernelCStr(kCopyKernel, "Copy");
  cl_context->setArg(0, x.storage());  // input
  cl_context->setArg(1, ret->storage());  // output
  uint32_t dim = 1;
  uint32_t nelem = x.nelems();
  cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);
  return ret;
}

template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::selectOuterDim(const Tensor<T>& src,
                                                     const uint32_t i) {
  RASSERT(src.dim_ != 0);  // Can't select on empty tensors.
  // For now, we don't support selecting scalars from vectors.
  RASSERT(src.dim_ > 1);
  RASSERT(i < src.size_[src.dim_ - 1]);

  std::shared_ptr<Tensor<T>> ret(new Tensor<T>());  // Empty tensor.

  // Calculate the return size.
  ret->dim_ = src.dim_ - 1;
  ret->size_.reset(new uint32_t[ret->dim_]);

  // Copy size from dims 0 to (dim - 1).
  for (uint32_t i = 0; i < ret->dim_; i++) {
    ret->size_[i] = src.size_[i];
  }

  // Calculate the offset in memory.
  std::unique_ptr<uint32_t[]> src_stride = src.calcStride();
  uint32_t offset = src_stride[src.dim_ - 1] * i;

  // Calculate the new size size.
  const uint32_t new_nelems = ret->nelems();

  ret->storage_ = src.storage_->createSubBuffer(new_nelems, offset);

  return ret;
}

template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::narrowOuterDim(const Tensor<T>& src,
                                                     const uint32_t i,
                                                     const uint32_t length) {
  RASSERT(length > 0); // Otherwise the output would be empty.
  RASSERT(src.dim_ != 0);  // Can't narrow on empty tensors.
  // For now, we don't support narrowing scalars from vectors.
  RASSERT(src.dim_ > 1 || length > 1);
  RASSERT(i < src.size_[src.dim_ - 1]);  // Make sure the start index fits.
  // Make sure the whole chunk fits.
  RASSERT(i + length - 1 < src.size_[src.dim_ - 1]);

  std::shared_ptr<Tensor<T>> ret(new Tensor<T>());  // Empty tensor.

  // Calculate the return size.
  ret->dim_ = src.dim_;
  ret->size_.reset(new uint32_t[ret->dim_]);

  // Copy size, and set the outputer dimension to length.
  for (uint32_t i = 0; i < ret->dim_; i++) {
    ret->size_[i] = src.size_[i];
  }
  ret->size_[ret->dim_ - 1] = length;

  // Calculate the offset in memory.
  std::unique_ptr<uint32_t[]> src_stride = src.calcStride();
  uint32_t offset = src_stride[src.dim_ - 1] * i;

  // Calculate the new size size.
  const uint32_t new_nelems = ret->nelems();

  ret->storage_ = src.storage_->createSubBuffer(new_nelems, offset);

  return ret;
}

template <typename T>
void Tensor<T>::copy(Tensor<T>& dst, const Tensor<T>& src) {
  RASSERT(src.dim_ != 0);
  RASSERT(dst.dim_ != 0);
  cl_context->useKernelCStr(kCopyKernel, "Copy");
  cl_context->setArg(0, src.storage());  // input
  cl_context->setArg(1, dst.storage());  // output
  uint32_t dim = 1;
  uint32_t nelem = dst.nelems();
  cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);
}

template <typename T>
void Tensor<T>::add(Tensor<T>& dst, const Tensor<T>& x, const Tensor<T>& y) {
  RASSERT(dst.dim_ != 0);
  RASSERT(x.dim_ != 0);
  RASSERT(y.dim_ != 0);
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
  RASSERT(dst.dim_ != 0);
  RASSERT(x.dim_ != 0);
  RASSERT(y.dim_ != 0);
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
  RASSERT(x.dim_ != 0);
  cl_context->useKernelCStr(kAbsKernel, "Abs");
  cl_context->setArg(0, x.storage());
  uint32_t dim = 1;
  uint32_t nelem = x.nelems();
  cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);
}

template <typename T>
void Tensor<T>::mul(Tensor<T>& x, float mul_val) {
  RASSERT(x.dim_ != 0);
  cl_context->useKernelCStr(kMulKernel, "Mul");
  cl_context->setArg(0, mul_val);
  cl_context->setArg(1, x.storage());
  uint32_t dim = 1;
  uint32_t nelem = x.nelems();
  cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);
}

template <typename T>
void Tensor<T>::div(Tensor<T>& x, float div_val) {
  RASSERT(x.dim_ != 0);
  cl_context->useKernelCStr(kMulKernel, "Mul");
  cl_context->setArg(0, 1.0f / div_val);
  cl_context->setArg(1, x.storage());
  uint32_t dim = 1;
  uint32_t nelem = x.nelems();
  cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);
}

template <typename T>
void Tensor<T>::add(Tensor<T>& x, float add_val) {
  RASSERT(x.dim_ != 0);
  cl_context->useKernelCStr(kAddScalarKernel, "AddScalarKernel");
  cl_context->setArg(0, add_val);
  cl_context->setArg(1, x.storage());
  uint32_t dim = 1;
  uint32_t nelem = x.nelems();
  cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);
}

template <typename T>
void Tensor<T>::accumulate(Tensor<T>& dst, const Tensor<T>& src) {
  RASSERT(src.dim_ != 0);
  RASSERT(dst.dim_ != 0);
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
  RASSERT(dst.dim_ != 0);
  cl_context->useKernelCStr(kFillKernel, "Fill");
  cl_context->setArg(0, dst.storage());
  cl_context->setArg(1, value);
  uint32_t dim = 1;
  uint32_t nelem = dst.nelems();
  cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);
}

template <typename T>
float Tensor<T>::slowSum(const Tensor<T>& x) {
  RASSERT(x.dim_ != 0);
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
  RASSERT(x.dim_ != 0);
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
  RASSERT(x.dim_ != 0);
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
float Tensor<T>::slowMean(const Tensor<T>& x) {
  RASSERT(x.dim_ != 0);
  std::unique_ptr<float[]> temp(new float[x.nelems()]);
  x.getData(temp.get());
  float mean = 0;
  for (uint32_t i = 0; i < x.nelems(); i++) {
    mean += temp[i];
  }
  return mean / static_cast<float>(x.nelems());
}

template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::loadFromFile(const std::string& file) {
  std::shared_ptr<Tensor<T>> new_tensor = nullptr;
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
    new_tensor = std::shared_ptr<Tensor<T>>(new Tensor<T>(dim, size));

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
void Tensor<T>::saveToFile(const Tensor<T>& tensor, const std::string& file) {
  std::ofstream ofile(file.c_str(), std::ios::out | std::ios::binary);
  if (ofile.is_open()) {
    // Now save the Tensor
    int32_t dim = tensor.dim_;
    ofile.write((char*)(&dim), sizeof(dim));
    for (int32_t i = dim - 1; i >= 0; i--) {
      int32_t cur_size = tensor.size_[i];
      ofile.write((char*)(&cur_size), sizeof(cur_size));
    }
    T* data = new T[tensor.nelems()];
    tensor.getData(data);
    ofile.write((char*)(data), sizeof(data[0]) * tensor.nelems());
    delete[] data;
    ofile.close();
  } else {
    std::cout << "Tensor<T>::saveToFile() - ERROR: Could not open file ";
    std::cout << file << std::endl;
    RASSERT(false);
  }
}

template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::slowRand(const uint32_t dim,
                                               const uint32_t* size) {
  RASSERT(dim > 0);

  std::shared_ptr<Tensor<T>> ret(new Tensor<T>(dim, size));
  const uint32_t nelems = ret->nelems();

  // Allocate the tensor on the CPU first.
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
  std::unique_ptr<float[]> ret_cpu(new float[nelems]);
  for (uint32_t i = 0; i < nelems; i++) {
    ret_cpu[i] = distribution(generator);
  }

  ret->setData(ret_cpu.get());
  return ret;
}

};  // namespace jtorch
