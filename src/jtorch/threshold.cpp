#include "jtorch/threshold.h"
#include "jtorch/jtorch.h"
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

  Threshold::Threshold() : TorchStage() {
    output = nullptr;
    threshold = 1e-6f;
    val = 0;
  }

  Threshold::~Threshold() {
    SAFE_DELETE(output);
  }

  void Threshold::init(TorchData& input)  {
    if (input.type() != TorchDataType::TENSOR_DATA) {
      throw std::runtime_error("Threshold::init() - "
        "FloatTensor expected!");
    }
    Tensor<float>& in = (Tensor<float>&)input;
    if (output != nullptr) {
      if (!in.isSameSizeAs(*TO_TENSOR_PTR(output))){
        // Input dimension has changed!
        delete output;
        output = nullptr;
      }
    }
    if (output == nullptr) {
      output = new Tensor<float>(in.dim(), in.size());
    }
  }

  void Threshold::forwardProp(TorchData& input) { 
    init(input);
    std::string kernel = jtorch::jtorch_path + "kernels/threshold.cl";
    cl_context->useKernel(kernel.c_str(), "Threshold1D");
    cl_context->setArg(0, ((Tensor<float>&)input).storage());
    cl_context->setArg(1, TO_TENSOR_PTR(output)->storage());
    cl_context->setArg(2, threshold);
    cl_context->setArg(3, val);
    uint32_t dim = 1;
    uint32_t nelem = TO_TENSOR_PTR(output)->nelems();
    cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);
  }

  TorchStage* Threshold::loadFromFile(std::ifstream& file) {
    Threshold* ret = new Threshold();
    file.read((char*)(&ret->threshold), sizeof(ret->threshold));
    file.read((char*)(&ret->val), sizeof(ret->val));
    return ret;
  }

}  // namespace jtorch