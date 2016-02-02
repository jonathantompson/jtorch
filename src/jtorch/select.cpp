#include "jtorch/select.h"

#include <cstring>

#include "jtorch/tensor.h"

using namespace jcl::threading;
using namespace jcl::math;

namespace jtorch {

Select::Select(int dimension, int index) : TorchStage() {
  dimension_ = dimension;
  index_ = index;
  src_tensor_ = nullptr;
}

Select::~Select() {}

void Select::forwardProp(std::shared_ptr<TorchData> input) {
  RASSERT(input->type() == TorchDataType::TENSOR_DATA);

  // For now we only support the Select operation along the outer dimension.
  // In torch indexing this is always 1.
  RASSERT(this->dimension_ == 1);

  if (src_tensor_ != input.get()) {
    // Only create the tensor slice if the input has changed.
    src_tensor_ = TO_TENSOR_PTR(input.get());

    // Note the index is torch 1-indexed.
    output = Tensor<float>::selectOuterDim(*src_tensor_, this->index_ - 1);
  }
}

std::unique_ptr<TorchStage> Select::loadFromFile(std::ifstream& file) {
  int dimension;
  file.read((char*)(&dimension), sizeof(dimension));
  int index;
  file.read(reinterpret_cast<char*>(&index), sizeof(index));
  std::unique_ptr<Select> ret(new Select(dimension, index));
  return std::unique_ptr<TorchStage>(std::move(ret));
}

}  // namespace jtorch
