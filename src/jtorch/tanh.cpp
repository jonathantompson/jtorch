#include "jtorch/tanh.h"
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

  Tanh::Tanh() : TorchStage() {
    output = nullptr;
  }

  Tanh::~Tanh() {
    SAFE_DELETE(output);
  }

  void Tanh::init(TorchData& input)  {
    if (input.type() != TorchDataType::TENSOR_DATA) {
      throw std::runtime_error("Tanh::init() - FloatTensor expected!");
    }
    Tensor<float>& in = (Tensor<float>&)input;
    Tensor<float>* out = (Tensor<float>*)output;
    if (output != nullptr) {
      if (!out->isSameSizeAs(in)) {
        // Input dimension has changed!
        SAFE_DELETE(output);
      }
    }
    if (output == nullptr) {
      output = new Tensor<float>(in.dim(), in.size());
      //cl_context->getOptimalLocalWorkgroupSizes(deviceid, 
      //  TO_TENSOR_PTR(output)->dim(), local_worgroup_size);
    }
  }

  void Tanh::forwardProp(TorchData& input) { 
    init(input);
    std::string kernel = jtorch::jtorch_path + "kernels/tanh.cl";
    cl_context->useKernel(kernel.c_str(), "TanH1D");
    cl_context->setArg(0, ((Tensor<float>&)input).storage());
    cl_context->setArg(1, TO_TENSOR_PTR(output)->storage());
    uint32_t dim = 1;
    uint32_t nelem = TO_TENSOR_PTR(output)->nelems();
    cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);
  }

  TorchStage* Tanh::loadFromFile(std::ifstream& file) {
    // Nothing to do for Tanh
    return new Tanh();
  }

}  // namespace jtorch