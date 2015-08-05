#include "jtorch/spatial_contrastive_normalization.h"
#include "jtorch/spatial_subtractive_normalization.h"
#include "jtorch/spatial_divisive_normalization.h"
#include "jtorch/sequential.h"
#include "jtorch/tensor.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"
#include "jcl/data_str/vector_managed.h"

#define SAFE_DELETE(x) if (x != nullptr) { delete x; x = nullptr; }
#define SAFE_DELETE_ARR(x) if (x != nullptr) { delete[] x; x = nullptr; }

using namespace jcl::threading;
using namespace jcl::math;
using namespace jcl::data_str;

namespace jtorch {

  // kernel1d default is either TorchStage::gaussian1D<float>(n) or just a
  // vector of 1 values.
  SpatialContrastiveNormalization::SpatialContrastiveNormalization(
    const Tensor<float>* kernel, const float threshold) : TorchStage() {
    const Tensor<float>* cur_kernel;
    if (kernel) {
      if (kernel->dim() > 2) {
        throw std::runtime_error("SpatialSubtractiveNormalization() - ERROR: "
          "Averaging kernel must be 1D or 2D!");
      }
      if (kernel->size()[0] % 2 == 0 || 
        (kernel->dim() == 2 && kernel->size()[1] % 2 == 0)) {
        throw std::runtime_error("SpatialSubtractiveNormalization() - ERROR: "
          "Averaging kernel must have odd size!");
      }
      cur_kernel = kernel;
    } else {
      uint32_t dim = 1;
      uint32_t size = 7;
      Tensor<float>* kernel = new Tensor<float>(dim, &size);
      Tensor<float>::fill(*kernel, 1);
      cur_kernel = kernel;
    }

    network_ = new Sequential();
    network_->add(new SpatialSubtractiveNormalization(*cur_kernel));
    network_->add(new SpatialDivisiveNormalization(*cur_kernel, threshold));

    if (kernel == nullptr) {
      // remove temporarily allocated kernel (since sub-modules will store
      // their own copy).
      delete cur_kernel;
    }
  }

  SpatialContrastiveNormalization::~SpatialContrastiveNormalization() {
    SAFE_DELETE(network_);
  }

  void SpatialContrastiveNormalization::forwardProp(TorchData& input) {
    network_->forwardProp(input);
    output = network_->output;
  }

  TorchStage* SpatialContrastiveNormalization::loadFromFile(std::ifstream& file) {
    // This whole thing is a little wasteful.  I copy to GPU here, and then
    // I copy it back down in the constructor anyway...  But it's good enough
    // for now.
    int32_t kernel_size_2, kernel_size_1;  // kernel_size_1 is the inner dim
    file.read((char*)(&kernel_size_1), sizeof(kernel_size_1));
    file.read((char*)(&kernel_size_2), sizeof(kernel_size_2));
    Tensor<float>* kernel;
    if (kernel_size_2 > 1) {
      // The kernel is 2D
      uint32_t dim = 2;
      uint32_t size[2] = {kernel_size_1, kernel_size_2};
      kernel = new Tensor<float>(dim, size);
    } else {
      uint32_t dim = 1;
      uint32_t size[1] = {kernel_size_1};
      kernel = new Tensor<float>(dim, size);
    }
    float* kernel_cpu = new float[kernel->nelems()];
    file.read((char*)(kernel_cpu), kernel->nelems() * sizeof(*kernel_cpu));
    kernel->setData(kernel_cpu);
    float threshold;
    file.read((char*)(&threshold), sizeof(threshold));
    TorchStage* ret = new SpatialContrastiveNormalization(kernel, threshold);
    delete kernel;
    delete[] kernel_cpu;
    return ret;
  }

}  // namespace jtorch