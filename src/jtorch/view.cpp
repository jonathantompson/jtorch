#include "jtorch/view.h"

#include <cstring>

#include "jtorch/tensor.h"

using namespace jcl::threading;
using namespace jcl::math;

namespace jtorch {

View::View(const uint32_t dim, const uint32_t* size) : TorchStage() {
  odim_ = dim;
  osize_.reset(new uint32_t[odim_]);
  memcpy(osize_.get(), size, sizeof(osize_[0]) * odim_);
  output = nullptr;
}

View::~View() {}

uint32_t View::outNElem() const {
  if (odim_ == 0) {
    return 0;
  }
  uint32_t ret = 1;
  for (uint32_t i = 0; i < odim_; i++) {
    ret *= osize_[i];
  }
  return ret;
}

void View::init(std::shared_ptr<TorchData> input) {
  RASSERT(input->type() == TorchDataType::TENSOR_DATA);
  Tensor<float>* in = TO_TENSOR_PTR(input.get());

  int32_t nelems = outNElem();
  static_cast<void>(nelems);
  // Check the input size.
  RASSERT(in->nelems() == static_cast<uint32_t>(nelems));

  if (output != nullptr) {
    Tensor<float>* out = TO_TENSOR_PTR(output.get());
    if (out->storage() != in->storage()) {
      // The tensors don't share the same storage! Reinitialize the view.
      output = nullptr;
    }
  }

  if (output == nullptr) {
    output = Tensor<float>::view(*in, odim_, osize_.get());
  }
}

void View::forwardProp(std::shared_ptr<TorchData> input) {
  init(input);
  // Nothing to do.  init will initialize our tensor view that points to the
  // same storage as the input.
}

std::unique_ptr<TorchStage> View::loadFromFile(std::ifstream& file) {
  int32_t dim;
  file.read((char*)(&dim), sizeof(dim));
  std::unique_ptr<uint32_t[]> size(new uint32_t[dim]);
  for (int32_t i = 0; i < dim; i++) {
    int32_t cur_size;
    file.read((char*)(&cur_size), sizeof(cur_size));
    size[i] = cur_size;
  }
  return std::unique_ptr<TorchStage>(new View(dim, size.get()));
}

}  // namespace jtorch
