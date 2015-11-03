#include "jtorch/narrow.h"

#include <cstring>

#include "jtorch/tensor.h"

using namespace jcl::threading;
using namespace jcl::math;

namespace jtorch {

Narrow::Narrow(int dimension, int index, int length) : TorchStage() {
  dimension_ = dimension;
  index_ = index;
  length_ = length;
  src_tensor_ = nullptr;
}

Narrow::~Narrow() {}

void Narrow::forwardProp(std::shared_ptr<TorchData> input) {
  RASSERT(input->type() == TorchDataType::TENSOR_DATA);

  // For now we only support the Narrow operation along the outer dimension.
  // In torch indexing this is always 1.
  RASSERT(this->dimension_ == 1);

  if (src_tensor_ != input.get()) {
    // Only create the tensor slice if the input has changed.
    src_tensor_ = TO_TENSOR_PTR(input.get());

    // Note the index is torch 1-indexed.
    output = Tensor<float>::narrowOuterDim(*src_tensor_, this->index_ - 1,
                                           this->length_);
  }
}

std::unique_ptr<TorchStage> Narrow::loadFromFile(std::ifstream& file) {
  int dimension;
  file.read((char*)(&dimension), sizeof(dimension));
  int index;
  file.read(reinterpret_cast<char*>(&index), sizeof(index));
  int length;
  file.read(reinterpret_cast<char*>(&length), sizeof(length));
  std::unique_ptr<Narrow> ret(new Narrow(dimension, index, length));
  return std::unique_ptr<TorchStage>(std::move(ret));
}

}  // namespace jtorch
