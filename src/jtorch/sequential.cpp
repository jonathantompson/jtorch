#include "jtorch/sequential.h"

#include "jtorch/tensor.h"

using namespace jcl::threading;
using namespace jcl::math;

namespace jtorch {

Sequential::Sequential() { output = nullptr; }

Sequential::~Sequential() {}

void Sequential::add(std::unique_ptr<TorchStage> stage) {
  network_.push_back(std::move(stage));
}

TorchStage* Sequential::get(const uint32_t i) { return network_[i].get(); }

uint32_t Sequential::size() const { return (uint32_t)network_.size(); }

std::unique_ptr<TorchStage> Sequential::loadFromFile(std::ifstream& file) {
  int n_nodes;
  file.read(reinterpret_cast<char*>(&n_nodes), sizeof(n_nodes));
  std::unique_ptr<Sequential> ret(new Sequential());
  ret->network_.reserve(n_nodes);
  for (int32_t i = 0; i < n_nodes; i++) {
    ret->network_.push_back(TorchStage::loadFromFile(file));
  }
  return std::unique_ptr<TorchStage>(std::move(ret));
}

void Sequential::forwardProp(std::shared_ptr<TorchData> input) {
  RASSERT(network_.size() > 0);
  network_[0]->forwardProp(input);
  for (uint32_t i = 1; i < network_.size(); i++) {
    std::shared_ptr<TorchData> cur_input = network_[i - 1]->output;
    network_[i]->forwardProp(cur_input);
  }
  output = network_[network_.size() - 1]->output;
}

}  // namespace jtorch
