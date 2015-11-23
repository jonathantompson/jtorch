#include "jtorch/concat_table.h"

#include <cstring>

#include "jtorch/table.h"
#include "jtorch/tensor.h"

using namespace jcl::threading;
using namespace jcl::math;

namespace jtorch {

ConcatTable::ConcatTable() : TorchStage() {
  output.reset(new Table());
}

ConcatTable::~ConcatTable() {}

void ConcatTable::add(std::unique_ptr<TorchStage> stage) {
  network_.push_back(std::move(stage));
}

TorchStage* ConcatTable::get(const uint32_t i) { return network_[i].get(); }

uint32_t ConcatTable::size() const { return (uint32_t)network_.size(); }

void ConcatTable::forwardProp(std::shared_ptr<TorchData> input) {
  RASSERT(network_.size() > 0);  // Otherwise no work to do.

  RASSERT(output->type() == TorchDataType::TABLE_DATA);
  Table* out = (Table*)output.get();
  out->clear();

  // FPROP each sub-module.
  std::vector<Tensor<float>*> outputs;
  for (uint32_t i = 0; i < (uint32_t)network_.size(); i++) {
    network_[i]->forwardProp(input);
    out->add(network_[i]->output);
  }
}

std::unique_ptr<TorchStage> ConcatTable::loadFromFile(std::ifstream& file) {
  int n_nodes;
  file.read(reinterpret_cast<char*>(&n_nodes), sizeof(n_nodes));
  std::unique_ptr<ConcatTable> ret(new ConcatTable());
  ret->network_.reserve(n_nodes);
  for (int32_t i = 0; i < n_nodes; i++) {
    ret->network_.push_back(TorchStage::loadFromFile(file));
  }
  return std::unique_ptr<TorchStage>(std::move(ret));
}

}  // namespace jtorch
