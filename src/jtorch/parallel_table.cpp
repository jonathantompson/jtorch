#include "jtorch/parallel_table.h"

#include "jtorch/table.h"
#include "jtorch/tensor.h"

using namespace jcl::threading;
using namespace jcl::math;

namespace jtorch {

ParallelTable::ParallelTable() { output = nullptr; }

ParallelTable::~ParallelTable() {}

void ParallelTable::add(std::unique_ptr<TorchStage> stage) {
  network_.push_back(std::move(stage));
  output = nullptr;
}

const uint32_t ParallelTable::size() const { return (uint32_t)network_.size(); }

std::unique_ptr<TorchStage> ParallelTable::loadFromFile(std::ifstream& file) {
  int n_nodes;
  file.read(reinterpret_cast<char*>(&n_nodes), sizeof(n_nodes));
  std::unique_ptr<ParallelTable> ret(new ParallelTable());
  ret->network_.reserve(n_nodes);
  for (int32_t i = 0; i < n_nodes; i++) {
    ret->network_.push_back(TorchStage::loadFromFile(file));
  }
  return std::unique_ptr<TorchStage>(std::move(ret));
}

void ParallelTable::initOutput() {
  if (output == nullptr) {
    output.reset(new Table());
  }

  Table* out = (Table*)output.get();
  out->clear();
  for (uint32_t i = 0; i < network_.size(); i++) {
    out->add(network_[i]->output);
  }
}

void ParallelTable::forwardProp(std::shared_ptr<TorchData> input) {
  RASSERT(input->type() == TorchDataType::TABLE_DATA);

  Table* in = (Table*)input.get();
  // Make sure table size matches the number of parallel stages:
  RASSERT(in->tableSize() == network_.size());
  for (uint32_t i = 0; i < network_.size(); i++) {
    network_[i]->forwardProp((*in)(i));
  }
  initOutput();  // Init output just copies the pointers from the output
                 // of all the parallel stages and fills up a table with them
}

uint32_t ParallelTable::numBanks() const { return (uint32_t)network_.size(); }

TorchStage* ParallelTable::get(const uint32_t i) { return network_[i].get(); }

}  // namespace jtorch
