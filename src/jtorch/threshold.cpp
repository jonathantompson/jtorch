#include "jtorch/threshold.h"
#include "jtorch/jtorch.h"
#include "jtorch/tensor.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"
#include "jcl/data_str/vector_managed.h"

#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }

using namespace jcl::threading;
using namespace jcl::math;
using namespace jcl::data_str;

namespace jtorch {

  Threshold::Threshold() : TorchStage() {
    output = NULL;
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
    if (output != NULL) {
      if (!Int3::equal(in.dim(), ((Tensor<float>*)output)->dim())) {
        // Input dimension has changed!
        delete output;
        output = NULL;
      }
    }
    if (output == NULL) {
      output = new Tensor<float>(in.dim());
      //cl_context->getOptimalLocalWorkgroupSizes(deviceid, 
      //  ((Tensor<float>*)output)->dim(), local_worgroup_size);
    }
  }

  void Threshold::forwardProp(TorchData& input) { 
    init(input);
    std::string kernel = jtorch::jtorch_path + "kernels/threshold.cl";
    cl_context->useKernel(kernel.c_str(), "Threshold1D");
    cl_context->setArg(0, ((Tensor<float>&)input).data());
    cl_context->setArg(1, ((Tensor<float>*)output)->data());
    cl_context->setArg(2, threshold);
    cl_context->setArg(3, val);
    cl_context->runKernel1D(jtorch::deviceid, output->dataSize(),
      false);
  }

  TorchStage* Threshold::loadFromFile(std::ifstream& file) {
    Threshold* ret = new Threshold();
    file.read((char*)(&ret->threshold), sizeof(ret->threshold));
    file.read((char*)(&ret->val), sizeof(ret->val));
    return ret;
  }

}  // namespace jtorch