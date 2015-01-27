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

#define JTORCH_TENSOR_PRECISON 4

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
    void getData(T* data) const;

    const jcl::math::Int3& dim() const { return dim_; }

    // Print --> EXPENSIVE
    virtual void print();  // print to std::cout
    void print(const jcl::math::Int2& interval0, 
      const jcl::math::Int2& interval1, 
      const jcl::math::Int2& interval2);

    // Some simple tensor math operations
    static void copy(Tensor<T>& dst, const Tensor<T>& src);
    static void add(Tensor<T>& dst, const Tensor<T>& x, const Tensor<T>& y);
    static void mul(Tensor<T>& x, float mul_value);
    static void div(Tensor<T>& x, float div_value);
    static void accumulate(Tensor<T>& dst, const Tensor<T>& src);
    static void zero(Tensor<T>& x);
    // slowSum - This does a CPU copy because I haven't written a reduction 
    // operator yet
    static float slowSum(const Tensor<T>& x);  
    
    // Some tensor math operations that return new tensors
    static Tensor<T>* clone(const Tensor<T>& x);
    static Tensor<T>* gaussian1D(const int32_t kernel_size);  // sigma = size / 2
    static Tensor<T>* gaussian(const int32_t kernel_size);
    static Tensor<T>* ones1D(const int32_t kernel_size);
    

    inline const jcl::JCLBuffer& data() const { return data_; }
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
    zero(*this);

  }

  template <typename T>
  Tensor<T>::Tensor(const jcl::math::Int2& dim) {
    dim_.set(dim[0], dim[1], 1);
    data_ = jtorch::cl_context->allocateBuffer(jcl::CLBufferTypeReadWrite,
      dim_[0], dim_[1], dim_[2]);
    zero(*this);
  }

  template <typename T>
  Tensor<T>::Tensor(const int dim) {
    dim_.set(dim, 1, 1);
    data_ = jtorch::cl_context->allocateBuffer(jcl::CLBufferTypeReadWrite,
      dim_[0], dim_[1], dim_[2]);
    zero(*this);
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
  void Tensor<T>::getData(T* data) const {
    jtorch::cl_context->readFromBuffer(data, jtorch::deviceid, data_, true);
  }

  template <typename T>
  void Tensor<T>::print() {
    std::streamsize prec = std::cout.precision();
    std::cout.precision(JTORCH_TENSOR_PRECISON);
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
          std::cout << std::fixed << data[v * dim_[0] + u];
          std::cout.unsetf(std::ios_base::floatfield);
          if (u != dim_[0] - 1) {
            std::cout << ", ";
          } else {
            std::cout << std::endl;
          }
        }
      }
    }
    std::cout.precision(prec);
    std::cout << std::resetiosflags(std::ios_base::showpos);
    delete[] d;

    std::cout << "[jtorch.";
    std::cout << jcl::JCL::CLDeviceToString(jtorch::cl_context->device());
    std::cout << " of dimension " << dim_[2] << "x" << dim_[1] << "x";
    std::cout << dim_[0] << "]" << std::endl;
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
    std::streamsize prec = std::cout.precision();
    std::cout.precision(JTORCH_TENSOR_PRECISON);
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
          std::cout << std::fixed << data[v * dim_[0] + u];
          std::cout.unsetf(std::ios_base::floatfield);
          if (u != interval0[1]) {
            std::cout << ", ";
          } else {
            std::cout << std::endl;
          }
        }
      }
    }
    std::cout.precision(prec);
    std::cout << std::resetiosflags(std::ios_base::showpos);
    delete[] d;

    std::cout << "[jtorch.";
    std::cout << jcl::JCL::CLDeviceToString(jtorch::cl_context->device());
    std::cout << " of dimension " << dim_[2] << "x" << dim_[1] << "x";
    std::cout << dim_[0] << "]" << std::endl;
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
  Tensor<T>* Tensor<T>::gaussian(const int32_t kernel_size) {
    Tensor<T>* ret = new Tensor<T>(jcl::math::Int3(kernel_size, kernel_size, 1));
    const float sigma = 0.25f;
    const float amplitude = 1.0f;
    const float size = (float)kernel_size;
    const float center = size/2.0f + 0.5f;
    T* data = new T[kernel_size * kernel_size];
    for (int32_t v = 0; v < kernel_size; v++) {
      for (int32_t u = 0; u < kernel_size; u++) {
        float du = ((float)(u+1) - center) / (sigma*size);
        float dv = ((float)(v+1) - center) / (sigma*size);
        data[v * kernel_size + u] = (T)amplitude * expf(-(du * du + dv * dv) / 
          2.0f);
      }
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
  Tensor<T>* Tensor<T>::clone(const Tensor<T>& x) {
    Tensor<T>* ret = new Tensor<T>(x.dim_);
    std::string kernel = jtorch::jtorch_path + "kernels/copy.cl";
    cl_context->useKernel(kernel.c_str(), "Copy");
    cl_context->setArg(0, x.data());
    cl_context->setArg(1, ret->data());
    cl_context->runKernel1D(jtorch::deviceid, ret->dataSize(), false);
    return ret;
  }
  
  template <typename T>
  void Tensor<T>::copy(Tensor<T>& dst, const Tensor<T>& src) {
    std::string kernel = jtorch::jtorch_path + "kernels/copy.cl";
    cl_context->useKernel(kernel.c_str(), "Copy");
    cl_context->setArg(0, src.data());
    cl_context->setArg(1, dst.data());
    cl_context->runKernel1D(jtorch::deviceid, dst.dataSize(), false);
  }

  template <typename T>
  void Tensor<T>::add(Tensor<T>& dst, const Tensor<T>& x, const Tensor<T>& y) {
    std::string kernel = jtorch::jtorch_path + "kernels/add.cl";
    cl_context->useKernel(kernel.c_str(), "Add");
    cl_context->setArg(0, src1.data());
    cl_context->setArg(1, src2.data());
    cl_context->setArg(2, dst.data());
    cl_context->runKernel1D(jtorch::deviceid, dst.dataSize(), false);
  }

  template <typename T>
  void Tensor<T>::mul(Tensor<T>& x, float mul_val) {
    std::string kernel = jtorch::jtorch_path + "kernels/mul.cl";
    cl_context->useKernel(kernel.c_str(), "Mul");
    cl_context->setArg(0, mul_val);
    cl_context->setArg(1, x.data());
    cl_context->runKernel1D(jtorch::deviceid, x.dataSize(), false);
  }

  template <typename T>
  void Tensor<T>::div(Tensor<T>& x, float div_val) {
    std::string kernel = jtorch::jtorch_path + "kernels/div.cl";
    cl_context->useKernel(kernel.c_str(), "Div");
    cl_context->setArg(0, div_val);
    cl_context->setArg(1, x.data());
    cl_context->runKernel1D(jtorch::deviceid, x.dataSize(), false);
  }

  template <typename T>
  void Tensor<T>::accumulate(Tensor<T>& dst, const Tensor<T>& src) {
    std::string kernel = jtorch::jtorch_path + "kernels/accumulate.cl";
    cl_context->useKernel(kernel.c_str(), "Accumulate");
    cl_context->setArg(0, src.data());
    cl_context->setArg(1, dst.data());
    cl_context->runKernel1D(jtorch::deviceid, dst.dataSize(), false);
  }

  template <typename T>
  void Tensor<T>::zero(Tensor<T>& dst) {
    std::string kernel = jtorch::jtorch_path + "kernels/zero.cl";
    cl_context->useKernel(kernel.c_str(), "Zero");
    cl_context->setArg(0, dst.data());
    cl_context->runKernel1D(jtorch::deviceid, dst.dataSize(), false);
  }

  template <typename T>
  float Tensor<T>::slowSum(const Tensor<T>& x) {
    float* temp = new float[x.dataSize()];
    x.getData(temp);
    float sum = 0.0f;
    for (uint32_t i = 0; i < x.dataSize(); i++) {
      sum += temp[i];
    }
    delete[] temp;
    return sum;
  }

};  // namespace jtorch
