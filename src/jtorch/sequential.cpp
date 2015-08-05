#include "jtorch/sequential.h"
#include "jtorch/tensor.h"
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

  Sequential::Sequential() {
    // Create an empty container
    network_ = new VectorManaged<TorchStage*>(1);
    output = nullptr;
  }

  Sequential::~Sequential() {
    SAFE_DELETE(network_);
  }

  void Sequential::add(TorchStage* stage) {
    network_->pushBack(stage);
  }

  TorchStage* Sequential::get(const uint32_t i) {
    return (*network_)[i];
  }

  uint32_t Sequential::size() const { 
    return network_->size();
  }

  TorchStage* Sequential::loadFromFile(std::ifstream& file) {
    int n_nodes;
    file.read(reinterpret_cast<char*>(&n_nodes), sizeof(n_nodes));
    Sequential* ret = new Sequential();
    ret->network_->capacity(n_nodes);
    for (int32_t i = 0; i < n_nodes; i++) {
      ret->network_->pushBack(TorchStage::loadFromFile(file));
    }
    return ret;
  }

  void Sequential::forwardProp(TorchData& input) {
    if (network_ == nullptr) {
      throw std::runtime_error("Sequential::forwardProp() - ERROR: "
        "Network is empty!");
    }
    (*network_)[0]->forwardProp(input);
    for (uint32_t i = 1; i < network_->size(); i++) {
      TorchData* cur_input = (*network_)[i-1]->output;
      (*network_)[i]->forwardProp(*cur_input);
    }
    output = (*network_)[network_->size()-1]->output;
  }

}  // namespace jtorch