#include "jtorch/spatial_contrastive_normalization.h"
#include "jtorch/spatial_subtractive_normalization.h"
#include "jtorch/spatial_divisive_normalization.h"
#include "jtorch/sequential.h"
#include "jtorch/tensor.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"
#include "jcl/data_str/vector_managed.h"

#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }

using namespace jcl::threading;
using namespace jcl::math;
using namespace jcl::data_str;

namespace jtorch {

  // kernel1d default is either TorchStage::gaussian1D<float>(n) or just a
  // vector of 1 values.
  SpatialContrastiveNormalization::SpatialContrastiveNormalization(
    const Tensor<float>* kernel1d, const float threshold) : TorchStage() {
    const Tensor<float>* kernel;
    if (kernel1d) {
      if (kernel1d->dataSize() % 2 == 0 || kernel1d->dim()[1] != 1 ||
        kernel1d->dim()[2] != 1) {
        throw std::runtime_error("SpatialSubtractiveNormalization() - ERROR: "
          "Averaging kernel must be 1D and have odd size!");
      }
      kernel = kernel1d;
    } else {
      kernel = Tensor<float>::ones1D(7);
    }

    network_ = new Sequential();
    network_->add(new SpatialSubtractiveNormalization(*kernel));
    network_->add(new SpatialDivisiveNormalization(*kernel, threshold));

    if (kernel1d == NULL) {
      // remove temporarily allocated kernel (since sub-modules will store
      // their own copy).
      delete kernel;
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
    int32_t kernel_size;
    file.read((char*)(&kernel_size), sizeof(kernel_size));
    Tensor<float>* kernel = new Tensor<float>(kernel_size);
    float* kernel_cpu = new float[kernel_size];
    file.read((char*)(kernel_cpu), kernel_size * sizeof(*kernel_cpu));
    kernel->setData(kernel_cpu);
    float threshold;
    file.read((char*)(&threshold), sizeof(threshold));
    TorchStage* ret = new SpatialContrastiveNormalization(kernel, threshold);
    delete kernel;
    delete[] kernel_cpu;
    return ret;
  }

}  // namespace jtorch