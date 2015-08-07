#include "jtorch/spatial_contrastive_normalization.h"
#include "jtorch/spatial_subtractive_normalization.h"
#include "jtorch/spatial_divisive_normalization.h"
#include "jtorch/sequential.h"
#include "jtorch/tensor.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"

using namespace jcl::threading;
using namespace jcl::math;

namespace jtorch {

// kernel1d default is either TorchStage::gaussian1D<float>(n) or just a
// vector of 1 values.
SpatialContrastiveNormalization::SpatialContrastiveNormalization(
    std::shared_ptr<Tensor<float>> kernel, float threshold)
    : TorchStage() {
  if (kernel) {
    // Averaging kernel must be 1D or 2D.
    RASSERT(kernel->dim() <= 2);

    // Averaging kernel must have odd size.
    RASSERT(kernel->size()[0] % 2 != 0 &&
           !(kernel->dim() == 2 && kernel->size()[1] % 2 == 0));
  } else {
    uint32_t dim = 1;
    uint32_t size = 7;
    kernel.reset(new Tensor<float>(dim, &size));
    Tensor<float>::fill(*kernel.get(), 1);
  }

  network_.reset(new Sequential());
  network_->add(
      std::unique_ptr<TorchStage>(new SpatialSubtractiveNormalization(kernel)));
  network_->add(std::unique_ptr<TorchStage>(
      new SpatialDivisiveNormalization(kernel, threshold)));
}

SpatialContrastiveNormalization::~SpatialContrastiveNormalization() {}

void SpatialContrastiveNormalization::forwardProp(
    std::shared_ptr<TorchData> input) {
  network_->forwardProp(input);
  output = network_->output;
}

std::unique_ptr<TorchStage> SpatialContrastiveNormalization::loadFromFile(
    std::ifstream& file) {
  // This whole thing is a little wasteful.  I copy to GPU here, and then
  // I copy it back down in the constructor anyway...  But it's good enough
  // for now.
  int32_t kernel_size_2, kernel_size_1;  // kernel_size_1 is the inner dim
  file.read((char*)(&kernel_size_1), sizeof(kernel_size_1));
  file.read((char*)(&kernel_size_2), sizeof(kernel_size_2));
  std::shared_ptr<Tensor<float>> kernel = nullptr;
  if (kernel_size_2 > 1) {
    // The kernel is 2D
    uint32_t dim = 2;
    uint32_t size[2] = {static_cast<uint32_t>(kernel_size_1),
                        static_cast<uint32_t>(kernel_size_2)};
    kernel.reset(new Tensor<float>(dim, size));
  } else {
    uint32_t dim = 1;
    uint32_t size[1] = {static_cast<uint32_t>(kernel_size_1)};
    kernel.reset(new Tensor<float>(dim, size));
  }
  std::unique_ptr<float[]> kernel_cpu(new float[kernel->nelems()]);
  file.read((char*)(kernel_cpu.get()),
            kernel->nelems() * sizeof(kernel_cpu.get()[0]));
  kernel->setData(kernel_cpu.get());
  float threshold;
  file.read((char*)(&threshold), sizeof(threshold));
  return std::unique_ptr<TorchStage>(
      new SpatialContrastiveNormalization(kernel, threshold));
}

}  // namespace jtorch
