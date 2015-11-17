#include "jtorch/spatial_dropout.h"

#include <cstring>

#include "jtorch/tensor.h"
#include "jtorch/jtorch.h"

using namespace jcl::threading;
using namespace jcl::math;
using namespace jcl;

namespace jtorch {

SpatialDropout::SpatialDropout(const float p) : TorchStage() {
  output = nullptr;
  p_ = p;
}

SpatialDropout::~SpatialDropout() {}

void SpatialDropout::init(std::shared_ptr<TorchData> input) {
  RASSERT(input->type() == TorchDataType::TENSOR_DATA);

  Tensor<float>* in = TO_TENSOR_PTR(input.get());
  if (output != nullptr) {
    if (!TO_TENSOR_PTR(output.get())->isSameSizeAs(*in)) {
      output = nullptr;
    }
  }
  if (output == nullptr) {
    output = Tensor<float>::clone(*in);
  }
}

void SpatialDropout::forwardProp(std::shared_ptr<TorchData> input) {
  init(input);

  Tensor<float>::copy(*TO_TENSOR_PTR(output.get()),
                      *TO_TENSOR_PTR(input.get()));
  Tensor<float>::mul(*TO_TENSOR_PTR(output.get()), 1 - p_);
}

std::unique_ptr<TorchStage> SpatialDropout::loadFromFile(std::ifstream& file) {
  float p;
  file.read((char*)(&p), sizeof(p));
  return std::unique_ptr<TorchStage>(new SpatialDropout(p));
}

}  // namespace jtorch
