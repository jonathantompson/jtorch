#include "jtorch/parallel_table.h"
#include "jtorch/tensor.h"
#include "jtorch/table.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"
#include "jcl/data_str/vector_managed.h"

#define SAFE_DELETE(x) if (x != nullptr) { delete x; x = nullptr; }
#define SAFE_DELETE_ARR(x) if (x != nullptr) { delete[] x; x = nullptr; }

using namespace jcl::threading;
using namespace jcl::math;
using namespace jcl::data_str;

namespace jtorch {

  ParallelTable::ParallelTable() {
    // Create an empty container
    network_ = new VectorManaged<TorchStage*>(1);
    output = nullptr;
  }

  ParallelTable::~ParallelTable() {
    SAFE_DELETE(network_);
    if (output != nullptr) {
      Table* out = (Table*)output;
      out->clearNoDelete();  // Remove the pointers without freeing memory
                             // Since they don't belong to this table.
    }
    SAFE_DELETE(output);
  }

  void ParallelTable::add(TorchStage* stage) {
    network_->pushBack(stage);
    output = nullptr;
  }

  const uint32_t ParallelTable::size() const {
    return network_->size();
  }

  TorchStage* ParallelTable::loadFromFile(std::ifstream& file) {
    int n_nodes;
    file.read(reinterpret_cast<char*>(&n_nodes), sizeof(n_nodes));
    ParallelTable* ret = new ParallelTable();
    ret->network_->capacity(n_nodes);
    for (int32_t i = 0; i < n_nodes; i++) {
      ret->network_->pushBack(TorchStage::loadFromFile(file));
    }
    return ret;
  }

  void ParallelTable::initOutput() {
    if (output == nullptr) {
      output = new Table();
    }

    Table* out = (Table*)output;
    out->clearNoDelete();
    for (uint32_t i = 0; i < network_->size(); i++) {
      out->add((*network_)[i]->output);
    }
  }

  void ParallelTable::forwardProp(TorchData& input) {
    if (input.type() != TorchDataType::TABLE_DATA) {
      throw std::runtime_error("Parallel::forwardProp() - "
        "Table expected!");
    }
    Table& in = (Table&)input;
    if (in.tableSize() != network_->size()) {
      throw std::runtime_error("Parallel::forwardProp() - ERROR: "
        "Table size does not match number of parallel stages!");
    }
    for (uint32_t i = 0; i < network_->size(); i++) {
      (*network_)[i]->forwardProp(*in(i));
    }
    initOutput();  // Init output just copies the pointers from the output
                   // of all the parallel stages and fills up a table with them
  }

  uint32_t ParallelTable::numBanks() const {
    if (network_ == nullptr) {
      throw std::runtime_error("Parallel::output() - ERROR: "
        "Network is empty!");
    }
    return (*network_).size();
  }

  TorchStage* ParallelTable::get(const uint32_t i) {
    if (network_ == nullptr) {
      throw std::runtime_error("Parallel::output() - ERROR: "
        "Network is empty!");
    }
    return (*network_)[i];
  }

}  // namespace jtorch