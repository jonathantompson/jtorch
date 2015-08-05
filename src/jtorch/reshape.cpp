#include "jtorch/reshape.h"
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

  Reshape::Reshape(const uint32_t dim, const uint32_t* size) : TorchStage() {
    odim_ = dim;
    osize_ = new uint32_t[odim_];
    memcpy(osize_, size, sizeof(osize_[0]) * odim_);
    output = nullptr;
  }

  Reshape::~Reshape() {
    SAFE_DELETE_ARR(osize_);
    SAFE_DELETE(output);
  }

  uint32_t Reshape::outNElem() const {
    if (odim_ == 0) {
      return 0;
    }
    uint32_t ret = 1;
    for (uint32_t i = 0; i < odim_; i++) {
      ret *= osize_[i];
    }
    return ret;
  }

  void Reshape::init(TorchData& input)  { 
    if (input.type() != TorchDataType::TENSOR_DATA) {
      throw std::runtime_error("Reshape::init() - "
        "FloatTensor expected!");
    }
    Tensor<float>& in = (Tensor<float>&)input;

    int32_t nelems = outNElem();
    if (in.nelems() != nelems) {
      throw std::runtime_error("Reshape::init() - Bad input size!");
    }

    if (output != nullptr) {
      Tensor<float>* out = (Tensor<float>*)output;
      if (out->storage() != in.storage()) {
        // The tensors don't share the same storage! Reinitialize the view.
        SAFE_DELETE(output);
      }
    }

    if (output == nullptr) {
      output = in.view(odim_, osize_);  // rets header that uses same storage
    }
  }

  void Reshape::forwardProp(TorchData& input) { 
    init(input);
    // Nothing to do.  init will initialize our tensor view that points to the
    // same storage as the input.
  }

  TorchStage* Reshape::loadFromFile(std::ifstream& file) {
    int32_t dim;
    file.read((char*)(&dim), sizeof(dim));
    uint32_t* size = new uint32_t[dim];
    for (int32_t i = 0; i < dim; i++) {
      int32_t cur_size;
      file.read((char*)(&cur_size), sizeof(cur_size));
      size[i] = cur_size;
    }
    TorchStage* stage = new Reshape(dim, size);
    SAFE_DELETE_ARR(size);
    return stage;
  }

}  // namespace jtorch