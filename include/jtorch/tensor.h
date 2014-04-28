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

#include <iostream>
#include <iomanip>
#include <fstream>
#include "jcl/math/int_types.h"
#include "jcl/math/math_types.h"
#include "jcl/jcl.h"  // For jcl::JCLBuffer
#include "jtorch/torch_data.h"
#include "jtorch/jtorch.h"

namespace jcl { namespace threading { class ThreadPool; } }

namespace jtorch {
  
  template <typename T>
  class Tensor : public TorchData {
  public:
    // Constructor / Destructor
    Tensor(const jcl::math::Int3& dim);  // Assumes dim[3]=1
    Tensor(const jcl::math::Int2& dim);  // Assumes dim[3]=1, dim[2]=1
    Tensor(const int dim);  // Assumes dim[3]=1, dim[2]=1, dim[1]=1
    virtual ~Tensor();

    virtual TorchDataType type() const { return TENSOR_DATA; }

    // setData and getData are EXPENSIVE --> They require a CPU to GPU copy
    void setData(const T* data);
    void getData(T* data);

    const jcl::math::Int3& dim() const { return dim_; }

    // Print --> EXPENSIVE
    virtual void print();  // print to std::cout
    void print(const jcl::math::Int2& interval0, 
      const jcl::math::Int2& interval1, 
      const jcl::math::Int2& interval2);

    // Deep copy (not too expensive since it'll copy GPU mem to GPU mem)
    Tensor<T>* copy() const;
    void zeroTensor() const;

    // gaussian1D In torch: >> normkernel = image.gaussian1D(n)
    // It's a gaussian of x = -2*sigma to 2*sigma, where sigma = size / 2
    static Tensor<T>* gaussian1D(const int32_t kernel_size);
    static Tensor<T>* ones1D(const int32_t kernel_size);

    inline jcl::JCLBuffer data() { return data_; }
    inline uint32_t dataSize() const { return dim_[0]*dim_[1]*dim_[2]; }

  protected:
    jcl::JCLBuffer data_;  // Internal data
    jcl::math::Int3 dim_;  // dim_[0] is lowest contiguous dimension, 
                             // dim_[2] is highest dimension

    // Non-copyable, non-assignable.
    Tensor(Tensor&);
    Tensor& operator=(const Tensor&);
  };

  template <typename T>
  Tensor<T>::Tensor(const jcl::math::Int3& dim) {
    dim_.set(dim[0], dim[1], dim[2]);
    data_ = jtorch::cl_context->allocateBuffer(jcl::CLBufferTypeReadWrite,
      dim_[0], dim_[1], dim_[2]);
    zeroTensor();

  }

  template <typename T>
  Tensor<T>::Tensor(const jcl::math::Int2& dim) {
    dim_.set(dim[0], dim[1], 1);
    data_ = jtorch::cl_context->allocateBuffer(jcl::CLBufferTypeReadWrite,
      dim_[0], dim_[1], dim_[2]);
    zeroTensor();
  }

  template <typename T>
  Tensor<T>::Tensor(const int dim) {
    dim_.set(dim, 1, 1);
    data_ = jtorch::cl_context->allocateBuffer(jcl::CLBufferTypeReadWrite,
      dim_[0], dim_[1], dim_[2]);
    zeroTensor();
  }

  template <typename T>
  Tensor<T>::~Tensor() {
    // Nothing to do
  }

  template <typename T>
  void Tensor<T>::setData(const T* data) {
    jtorch::cl_context->writeToBuffer(data, jtorch::deviceid, data_, true);
  }

  template <typename T>
  void Tensor<T>::getData(T* data) {
    jtorch::cl_context->readFromBuffer(data, jtorch::deviceid, data_, true);
  }

  template <typename T>
  void Tensor<T>::print() {
    T* d = new T[dataSize()];
    getData(d);
    const int32_t dim = dim_[0] * dim_[1];
    for (int32_t i = 0; i < dim_[2]; i++) {
#if defined(WIN32) || defined(_WIN32)
      std::cout.setf(0, std::ios::showpos);
#else
      std::cout.setf(std::ios::showpos);
#endif
      std::cout << "  3dtensor[" << i << ", *, *] =";
      std::cout << std::endl;
      T* data = &d[i * dim_[1]*dim_[0]];
      for (int32_t v = 0; v < dim_[1]; v++) {
        if (v == 0) {
          std::cout << "    (0,0) ";
        } else {
          std::cout << "          ";
        }
        std::cout.setf(std::ios::showpos);
        for (int32_t u = 0; u < dim_[0]; u++) {
          std::cout << data[v * dim_[0] + u];
          if (u != dim_[0] - 1) {
            std::cout << ", ";
          } else {
            std::cout << std::endl;
          }
        }
      }
    }
    std::cout << std::resetiosflags(std::ios_base::showpos);
    delete[] d;
  };

  template <typename T>
  void Tensor<T>::print(const jcl::math::Int2& interval0, 
    const jcl::math::Int2& interval1, const jcl::math::Int2& interval2) {
    if (interval0[0] > interval0[1] || interval1[0] > interval1[1] || 
      interval2[0] > interval2[1]) {
      throw std::runtime_error("Tensor<T>::print() - ERROR: "
        "intervals must be monotonic");
    }
    if (interval0[0] < 0 || interval0[1] >= dim_[0] || 
      interval1[0] < 0 || interval1[1] >= dim_[1] ||
      interval2[0] < 0 || interval2[1] >= dim_[2]) {
      throw std::runtime_error("Tensor<T>::print() - ERROR: "
        "intervals out of range");
    }
    T* d = new T[dataSize()];
    getData(d);
    for (int32_t f = interval2[0]; f <= interval2[1]; f++) {
#if defined(WIN32) || defined(_WIN32)
      std::cout.setf(0, std::ios::showpos);
#else
      std::cout.setf(std::ios::showpos);
#endif
      std::cout << "  3dtensor[" << f << ", *, *] =";
      std::cout << std::endl;
      T* data = &d[f * dim_[1]*dim_[0]];
      for (int32_t v = interval1[0]; v <= interval1[1]; v++) {
        if (v == interval1[0]) {
          std::cout << "    (" << interval1[0] << "," <<  interval0[0] << ") ";
        } else {
          std::cout << "          ";
        }
        std::cout.setf(std::ios::showpos);
        for (int32_t u = interval0[0]; u <= interval0[1]; u++) {
          std::cout << data[v * dim_[0] + u];
          if (u != interval0[1]) {
            std::cout << ", ";
          } else {
            std::cout << std::endl;
          }
        }
      }
    }
    delete[] d;
  }

  template <typename T>
  Tensor<T>* Tensor<T>::gaussian1D(const int32_t kernel_size) {
    Tensor<T>* ret = new Tensor<T>(jcl::math::Int3(kernel_size, 1, 1));
    const float sigma = 0.25f;
    const float amplitude = 1.0f;
    const float size = (float)kernel_size;
    const float center = size/2.0f + 0.5f;
    T* data = new T[kernel_size];
    for (int32_t i = 0; i < kernel_size; i++) {
      data[i] = (T)amplitude * expf(-(powf(((float)(i+1) - center) / 
        (sigma*size), 2.0f) / 2.0f));
    }
    ret->setData(data);
    delete[] data;
    return ret;
  }

  template <typename T>
  Tensor<T>* Tensor<T>::ones1D(const int32_t kernel_size) {
    Tensor<T>* ret = new Tensor<T>(jcl::math::Int3(kernel_size, 1, 1));
    T* data = new T[kernel_size];
    for (int32_t i = 0; i < kernel_size; i++) {
      data[i] = (T)1;
    }
    ret->setData(data);
    delete[] data;
    return ret;
  }

  template <typename T>
  Tensor<T>* Tensor<T>::copy() const {
    Tensor<T>* ret = new Tensor<T>(dim_);
    std::string kernel = jtorch::jtorch_path + "kernels/tensor.cl";
    cl_context->useKernel(kernel.c_str(), "Copy");
    cl_context->setArg(0, const_cast<Tensor<T>*>(this)->data());
    cl_context->setArg(1, ret->data());
    cl_context->runKernel1D(jtorch::deviceid, ret->dataSize(), false);
    return ret;
  }
  
  template <typename T>
  void Tensor<T>::zeroTensor() const {
    std::string kernel = jtorch::jtorch_path + "kernels/tensor.cl";
    cl_context->useKernel(kernel.c_str(), "Zero");
    cl_context->setArg(0, const_cast<Tensor<T>*>(this)->data());
    cl_context->runKernel1D(jtorch::deviceid, this->dataSize(), false);
  }

};  // namespace jtorch
