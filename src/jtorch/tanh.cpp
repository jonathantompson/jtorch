#include "jtorch/tanh.h"
#include "jtorch/jtorch.h"
#include "jtorch/tensor.h"
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

  Tanh::Tanh() : TorchStage() {
    output = NULL;
  }

  Tanh::~Tanh() {
    SAFE_DELETE(output);
  }

  void Tanh::init(TorchData& input)  {
    if (input.type() != TorchDataType::TENSOR_DATA) {
      throw std::wruntime_error("Tanh::init() - FloatTensor expected!");
    }
    Tensor<float>& in = (Tensor<float>&)input;
    if (output != NULL) {
      if (!Int3::equal(in.dim(), ((Tensor<float>*)output)->dim())) {
        // Input dimension has changed!
        SAFE_DELETE(output);
      }
    }
    if (output == NULL) {
      output = new Tensor<float>(in.dim());
      //cl_context->getOptimalLocalWorkgroupSizes(deviceid, 
      //  ((Tensor<float>*)output)->dim(), local_worgroup_size);
    }
  }

  void Tanh::forwardProp(TorchData& input) { 
    init(input);
    std::string kernel = jtorch::jtorch_path + "kernels/tanh.cl";
    cl_context->useKernel(kernel.c_str(), "TanH1D");
    cl_context->setArg(0, ((Tensor<float>&)input).data());
    cl_context->setArg(1, ((Tensor<float>*)output)->data());
    cl_context->runKernel1D(jtorch::deviceid, output->dataSize(),
      false);
  }

  TorchStage* Tanh::loadFromFile(std::ifstream& file) {
    // Nothing to do for Tanh
    return new Tanh();
  }

}  // namespace jtorch