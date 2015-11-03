#include "jtorch/concat.h"

#include <cstring>

#include "jtorch/tensor.h"

using namespace jcl::threading;
using namespace jcl::math;

namespace jtorch {

Concat::Concat(int dimension) : TorchStage() {
  output.reset(new Tensor<float>());
  dimension_ = dimension;
}

Concat::~Concat() {}

void Concat::add(std::unique_ptr<TorchStage> stage) {
  network_.push_back(std::move(stage));
}

TorchStage* Concat::get(const uint32_t i) { return network_[i].get(); }

uint32_t Concat::size() const { return (uint32_t)network_.size(); }

void Concat::forwardProp(std::shared_ptr<TorchData> input) {
  RASSERT(network_.size() > 0);  // Otherwise no work to do.

  // For now we only support the Concat operation along the outer dimension.
  // In torch indexing this is always 1.
  RASSERT(this->dimension_ == 1);

  // FPROP each sub-module.
  std::vector<Tensor<float>*> outputs;
  for (uint32_t i = 0; i < (uint32_t)network_.size(); i++) {
    network_[i]->forwardProp(input);
    // Each output must be a tensor.
    RASSERT(network_[i]->output->type() == TorchDataType::TENSOR_DATA);
    outputs.push_back(TO_TENSOR_PTR(network_[i]->output.get()));
    // All tensors must be the same dimension.
    RASSERT(outputs[i]->dim() == outputs[0]->dim());
  }

  // Note the dimension from torch is 1-index with dimension 1 being the outer
  // dimension. This is the opposite from jtorch.
  const uint32_t dim = outputs[0]->dim();
  uint32_t concat_dim = dim - static_cast<uint32_t>(this->dimension_);

  // Check that all tensors are the same size except across the concat
  // dimension.
  for (uint32_t i = 1; i < outputs.size(); i++) {
    for (uint32_t j = 0; j < dim; j++) {
      if (j != concat_dim) {
        RASSERT(outputs[i]->size()[j] == outputs[0]->size()[j]);
      }
    }
  }

  // Calculate the size of the output tensor.
  std::unique_ptr<uint32_t[]> out_sz(new uint32_t[dim]);
  for (uint32_t i = 0; i < dim; i++) {
    out_sz[i] = outputs[0]->size()[i];
  }
  for (uint32_t i = 1; i < outputs.size(); i++) {
    out_sz[concat_dim] += outputs[i]->size()[concat_dim];
  }

  // Resize the output tensor if needed.
  Tensor<float>* out = TO_TENSOR_PTR(output.get());
  out->resize(dim, out_sz.get());

  // Now copy the outputs of the sub-modules into the output tensor.
  uint32_t f_offset = 0;
  for (uint32_t i = 0; i < (uint32_t)outputs.size(); i++) {
    const uint32_t f_size = outputs[i]->size()[concat_dim];
    // TODO(tompson): Worry here, this is allocating lots of tensors using
    // createSubBuffer, which may or may not be slow (driver dependent).
    // Instead we could pre-allocate these slices.
    std::shared_ptr<Tensor<float>> oslice =
        Tensor<float>::narrowOuterDim(*out, f_offset, f_size);
    f_offset += f_size;
    Tensor<float>::copy(*oslice, *(outputs[i]));
  }
}

std::unique_ptr<TorchStage> Concat::loadFromFile(std::ifstream& file) {
  int dimension;
  file.read((char*)(&dimension), sizeof(dimension));
  int n_nodes;
  file.read(reinterpret_cast<char*>(&n_nodes), sizeof(n_nodes));
  std::unique_ptr<Concat> ret(new Concat(dimension));
  ret->network_.reserve(n_nodes);
  for (int32_t i = 0; i < n_nodes; i++) {
    ret->network_.push_back(TorchStage::loadFromFile(file));
  }
  return std::unique_ptr<TorchStage>(std::move(ret));
}

}  // namespace jtorch
