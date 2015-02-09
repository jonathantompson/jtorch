#include "jtorch/spatial_dropout.h"
#include "jtorch/tensor.h"
#include "jtorch/jtorch.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"
#include "jcl/data_str/vector_managed.h"

#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }

using namespace jcl::threading;
using namespace jcl::math;
using namespace jcl::data_str;
using namespace jcl;

namespace jtorch {

  SpatialDropout::SpatialDropout(const float p) : TorchStage() {
    output = NULL;
    p_ = p;
  }

  SpatialDropout::~SpatialDropout() {
    SAFE_DELETE(output);
  }

  void SpatialDropout::init(TorchData& input)  {
    if (input.type() != TorchDataType::TENSOR_DATA) {
      throw std::runtime_error("SpatialDropout::init() - "
        "FloatTensor expected!");
    }
    Tensor<float>& in = (Tensor<float>&)input;
    if (output != NULL) {
      if (!TO_TENSOR_PTR(output)->isSameSizeAs((Tensor<float>&)input)) {
        SAFE_DELETE(output);
      }
    }
    if (output == NULL) {
      output = Tensor<float>::clone((Tensor<float>&)input);
    }
  }

  void SpatialDropout::forwardProp(TorchData& input) { 
    init(input);

    Tensor<float>::copy(*TO_TENSOR_PTR(output), (Tensor<float>&)input);
    Tensor<float>::mul(*TO_TENSOR_PTR(output), p_);
  }

  TorchStage* SpatialDropout::loadFromFile(std::ifstream& file) {
    float p;
    file.read((char*)(&p), sizeof(p));
    return new SpatialDropout(p);
  }

}  // namespace jtorch