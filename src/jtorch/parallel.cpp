#include "jtorch/parallel.h"
#include "jtorch/tensor.h"
#include "jtorch/table.h"
#include "jtil/exceptions/wruntime_error.h"
#include "jtil/threading/thread.h"
#include "jtil/threading/callback.h"
#include "jtil/threading/thread_pool.h"
#include "jtil/data_str/vector_managed.h"

#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }

using namespace jtil::threading;
using namespace jtil::math;
using namespace jtil::data_str;

namespace jtorch {

  Parallel::Parallel() {
    // Create an empty container
    network_ = new VectorManaged<TorchStage*>(1);
    output = NULL;
  }

  Parallel::~Parallel() {
    SAFE_DELETE(network_);
    if (output != NULL) {
      Table* out = (Table*)output;
      out->clearNoDelete();  // Remove the pointers without freeing memory
                             // Since they don't belong to this table.
    }
    SAFE_DELETE(output);
  }

  void Parallel::add(TorchStage* stage) {
    network_->pushBack(stage);
    output = NULL;
  }

  TorchStage* Parallel::loadFromFile(std::ifstream& file) {
    int n_nodes;
    file.read(reinterpret_cast<char*>(&n_nodes), sizeof(n_nodes));
    Parallel* ret = new Parallel();
    ret->network_->capacity(n_nodes);
    for (int32_t i = 0; i < n_nodes; i++) {
      ret->network_->pushBack(TorchStage::loadFromFile(file));
    }
    return ret;
  }

  void Parallel::initOutput() {
    if (output == NULL) {
      output = new Table();
    }

    Table* out = (Table*)output;
    out->clearNoDelete();
    for (uint32_t i = 0; i < network_->size(); i++) {
      out->add((Tensor<float>*)(*network_)[i]->output);
    }
  }

  void Parallel::forwardProp(TorchData& input) {
    if (input.type() != TorchDataType::TABLE_DATA) {
      throw std::wruntime_error("Parallel::forwardProp() - "
        "Table expected!");
    }
    Table& in = (Table&)input;
    if (in.tableSize() != network_->size()) {
      throw std::wruntime_error("Parallel::forwardProp() - ERROR: "
        "Table size does not match number of parallel stages!");
    }
    for (uint32_t i = 0; i < network_->size(); i++) {
      (*network_)[i]->forwardProp(*in(i));
    }
    initOutput();  // Init output just copies the pointers from the output
                   // of all the parallel stages and fills up a table with them
  }

  uint32_t Parallel::numBanks() const {
    if (network_ == NULL) {
      throw std::wruntime_error("Parallel::output() - ERROR: "
        "Network is empty!");
    }
    return (*network_).size();
  }

  TorchStage* Parallel::get(const uint32_t i) {
    if (network_ == NULL) {
      throw std::wruntime_error("Parallel::output() - ERROR: "
        "Network is empty!");
    }
    return (*network_)[i];
  }

}  // namespace jtorch