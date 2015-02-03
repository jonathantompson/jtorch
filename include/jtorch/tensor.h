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
#include <sstream>
#include "jcl/math/int_types.h"
#include "jcl/math/math_types.h"
#include "jcl/jcl.h"  // For jcl::JCLBuffer
#include "jtorch/torch_data.h"
#include "jtorch/jtorch.h"

#define JTORCH_TENSOR_PRECISON 4

namespace jcl { namespace threading { class ThreadPool; } }

#define TO_TENSOR_PTR(x) (x->type() == jtorch::TorchDataType::TENSOR_DATA ? (jtorch::Tensor<float>*)x : NULL)

namespace jtorch {
  
  template <typename T>
  class Tensor : public TorchData {
  public:
    Tensor(const uint32_t dim, const uint32_t* size);
    virtual ~Tensor();

    virtual TorchDataType type() const { return TENSOR_DATA; }

    // setData and getData are EXPENSIVE --> They require a CPU to GPU copy
    void setData(const T* data);
    void getData(T* data) const;

    // View returns a new view on the same object.  The caller owns the new
    // memory (ie, it is transferred).
    Tensor<T>* view(const uint32_t dim, const uint32_t* size);

    const uint32_t dim() const { return dim_; }
    const uint32_t* size() const { return size_; }
    const bool isSameSizeAs(const Tensor<T>& src) const;

    // Print --> EXPENSIVE
    virtual void print();  // print to std::cout

    // Some simple tensor math operations
    static void copy(Tensor<T>& dst, const Tensor<T>& src);
    static void add(Tensor<T>& dst, const Tensor<T>& x, const Tensor<T>& y);
    static void mul(Tensor<T>& x, float mul_value);
    static void div(Tensor<T>& x, float div_value);
    static void accumulate(Tensor<T>& dst, const Tensor<T>& src);
    static void zero(Tensor<T>& x);
    static void fill(Tensor<T>& x, float value);
    // slowSum - This does a CPU copy because I haven't written a reduction 
    // operator yet
    static float slowSum(const Tensor<T>& x);  
    
    // Some tensor math operations that return new tensors
    static Tensor<T>* clone(const Tensor<T>& x);
    static Tensor<T>* gaussian1D(const int32_t kernel_size);  // sigma = size / 2
    static Tensor<T>* gaussian(const int32_t kernel_size);
    static Tensor<T>* loadFromFile(const std::string& file);

    inline const jcl::JCLBuffer& storage() const { return storage_; }
    inline uint32_t nelems() const;

    uint32_t* calcStride() const;  // memory returned is owned by caller

  protected:
    jcl::JCLBuffer storage_;  // Internal data
    uint32_t dim_;
    uint32_t* size_;  // size_[0] is lowest contiguous dimension, 
                      // size_[2] is highest dimension

    Tensor();  // Default constructor used internally (in view function)

    // Non-copyable, non-assignable.
    Tensor(Tensor&);
    Tensor& operator=(const Tensor&);
  };

  template <typename T>
  Tensor<T>::Tensor(const uint32_t dim, const uint32_t* size) {
    this->dim_ = dim;
    this->size_ = new uint32_t[dim];
    memcpy(this->size_, size, sizeof(this->size_[0]) * dim);
    storage_ = jtorch::cl_context->allocateBuffer(jcl::CLBufferTypeReadWrite,
      nelems());
    zero(*this);

  }

  template <typename T>
  Tensor<T>::Tensor() {
    // Default constructor returns an empty header.  Used internally (ie 
    // private).
    dim_ = 0;
    size_ = NULL;
    storage_ = (jcl::JCLBuffer)-1;
  }

  template <typename T>
  Tensor<T>::~Tensor() {
    if (size_) {
      delete[] size_;
    }
  }

  template <typename T>
  uint32_t* Tensor<T>::calcStride() const {
    uint32_t* stride = new uint32_t[dim_];
    stride[0] = 1;
    for (uint32_t i = 1; i < dim_; i++) {
      stride[i] = stride[i-1] * size_[i-1];
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
  Tensor<T>* Tensor<T>::view(const uint32_t dim, const uint32_t* size) {
    if (dim == 0) {
      throw std::runtime_error("ERROR - view() - zero dimension not allowed!"); 
    }
    int32_t view_nelem = 1;
    for (uint32_t i = 0; i < dim; i++) {
      view_nelem *= size[i];
    }

    if (view_nelem != nelems()) {
      throw std::runtime_error("ERROR - view() - Size mismatch!"); 
    }

    Tensor<T>* return_header = new Tensor<T>();
    return_header->dim_ = dim;
    return_header->size_ = new uint32_t[dim];
    memcpy(return_header->size_, size, sizeof(return_header->size_[0]) * dim);
    return_header->storage_ = storage_;
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
    T scale = (T)pow(10.0, floor(log10((double)max_val + EPSILON)));

#if defined(WIN32) || defined(_WIN32)
      std::cout.setf(0, std::ios::showpos);
#else
      std::cout.setf(std::ios::showpos);
#endif

    if (dim_ == 1) {
      // Print a 1D tensor
      std::cout << "  tensor[*] =" << std::endl;
      if (fabsf((float)scale - 1.0f) > EPSILON) {
        std::cout << " " << scale << " * " << std::endl;
      }
      std::cout.setf(std::ios::showpos);
      for (uint32_t u = 0; u < size_[0]; u++) {
        if (u == 0) {
          std::cout << " (0) ";
        } else {
          std::cout << "     ";
        }
        std::cout << std::fixed << d[u] / scale << std::endl;;
        std::cout.unsetf(std::ios_base::floatfield);
      }
    } else if (dim_ == 2) {
      // Print a 2D tensor
      std::cout << "  tensor[*,*] =" << std::endl;
      if (fabsf((float)scale - 1.0f) > EPSILON) {
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
        for (uint32_t cur_dim = dim_-1; cur_dim >= 2; cur_dim--) {
          std::cout << (i % stride[cur_dim]) << ",";
        }

        std::cout << "*,*] =";
        std::cout << std::endl;
        if (fabsf((float)scale - 1.0f) > EPSILON) {
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
    for (int32_t i = (int32_t)dim_-1; i >= 0; i--) {
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
    const float center = (float)kernel_size/2.0f + 0.5f;
    T* data = new T[kernel_size];
    for (int32_t i = 0; i < kernel_size; i++) {
      data[i] = (T)amplitude * expf(-(powf(((float)(i+1) - center) / 
        (sigma*(float)kernel_size), 2.0f) / 2.0f));
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
    const float center = (float)kernel_size/2.0f + 0.5f;
    T* data = new T[kernel_size * kernel_size];
    for (int32_t v = 0; v < kernel_size; v++) {
      for (int32_t u = 0; u < kernel_size; u++) {
        float du = ((float)(u+1) - center) / (sigma*(float)kernel_size);
        float dv = ((float)(v+1) - center) / (sigma*(float)kernel_size);
        data[v * kernel_size + u] = (T)amplitude * expf(-(du * du + dv * dv) / 
          2.0f);
      }
    }
    ret->setData(data);
    delete[] data;
    return ret;
  }

  template <typename T>
  Tensor<T>* Tensor<T>::clone(const Tensor<T>& x) {
    Tensor<T>* ret = new Tensor<T>(x.dim_, x.size_);
    std::string kernel = jtorch::jtorch_path + "kernels/copy.cl";
    cl_context->useKernel(kernel.c_str(), "Copy");
    cl_context->setArg(0, x.storage());
    cl_context->setArg(1, ret->storage());
    uint32_t dim = 1;
    uint32_t nelem = x.nelems();
    cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);
    return ret;
  }
  
  template <typename T>
  void Tensor<T>::copy(Tensor<T>& dst, const Tensor<T>& src) {
    std::string kernel = jtorch::jtorch_path + "kernels/copy.cl";
    cl_context->useKernel(kernel.c_str(), "Copy");
    cl_context->setArg(0, src.storage());
    cl_context->setArg(1, dst.storage());
    uint32_t dim = 1;
    uint32_t nelem = dst.nelems();
    cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);
  }

  template <typename T>
  void Tensor<T>::add(Tensor<T>& dst, const Tensor<T>& x, const Tensor<T>& y) {
    std::string kernel = jtorch::jtorch_path + "kernels/add.cl";
    cl_context->useKernel(kernel.c_str(), "Add");
    cl_context->setArg(0, src1.storage());
    cl_context->setArg(1, src2.storage());
    cl_context->setArg(2, dst.storage());
    uint32_t dim = 1;
    uint32_t nelem = dst.nelems();
    cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);
  }

  template <typename T>
  void Tensor<T>::mul(Tensor<T>& x, float mul_val) {
    std::string kernel = jtorch::jtorch_path + "kernels/mul.cl";
    cl_context->useKernel(kernel.c_str(), "Mul");
    cl_context->setArg(0, mul_val);
    cl_context->setArg(1, x.storage());
    uint32_t dim = 1;
    uint32_t nelem = x.nelems();
    cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);
  }

  template <typename T>
  void Tensor<T>::div(Tensor<T>& x, float div_val) {
    std::string kernel = jtorch::jtorch_path + "kernels/div.cl";
    cl_context->useKernel(kernel.c_str(), "Div");
    cl_context->setArg(0, div_val);
    cl_context->setArg(1, x.storage());
    uint32_t dim = 1;
    uint32_t nelem = x.nelems();
    cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);
  }

  template <typename T>
  void Tensor<T>::accumulate(Tensor<T>& dst, const Tensor<T>& src) {
    std::string kernel = jtorch::jtorch_path + "kernels/accumulate.cl";
    cl_context->useKernel(kernel.c_str(), "Accumulate");
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
    std::string kernel = jtorch::jtorch_path + "kernels/fill.cl";
    cl_context->useKernel(kernel.c_str(), "Fill");
    cl_context->setArg(0, dst.storage());
    cl_context->setArg(1, value);
    uint32_t dim = 1;
    uint32_t nelem = dst.nelems();
    cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);
  }

  template <typename T>
  float Tensor<T>::slowSum(const Tensor<T>& x) {
    float* temp = new float[x.nelems()];
    x.getData(temp);
    float sum = 0.0f;
    for (uint32_t i = 0; i < x.nelems(); i++) {
      sum += temp[i];
    }
    delete[] temp;
    return sum;
  }

  template <typename T>
  Tensor<T>* Tensor<T>::loadFromFile(const std::string& file) {
    Tensor<T>* new_tensor = NULL;
    std::ifstream ifile(file.c_str(), std::ios::in|std::ios::binary);
    if (ifile.is_open()) {
      ifile.seekg(0, std::ios::beg);
      // Now load the Tensor
      int32_t dim;
      ifile.read((char*)(&dim), sizeof(dim));
      uint32_t* size = new uint32_t[dim];
      for (int32_t i = 0; i < dim; i++) {
        int32_t cur_size;
        ifile.read((char*)(&cur_size), sizeof(cur_size));
        size[dim-i-1] = (uint32_t)cur_size;
      }
      new_tensor = new Tensor<T>(dim, size);

      T* data = new T[new_tensor->nelems()];
      ifile.read((char*)(data), sizeof(data[0]) * new_tensor->nelems());
      new_tensor->setData(data);
      delete[] data;
      ifile.close();
      delete[] size;
    } else {
      std::stringstream ss;
      ss << "Tensor<T>::loadFromFile() - ERROR: Could not open file ";
      ss << file << std::endl;
      throw std::runtime_error(ss.str());
    }
    return new_tensor;
  }

};  // namespace jtorch
