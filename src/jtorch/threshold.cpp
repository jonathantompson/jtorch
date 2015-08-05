#include "jtorch/threshold.h"
#include "jtorch/jtorch.h"
#include "jtorch/tensor.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"
#include "jcl/data_str/vector_managed.h"

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
  }

  void Threshold::init(std::shared_ptr<TorchData> input)  {
    assert(input->type() == TorchDataType::TENSOR_DATA);
    Tensor<float>* in = TO_TENSOR_PTR(input.get());
    if (output != nullptr) {
      if (!in->isSameSizeAs(*TO_TENSOR_PTR(output.get()))){
        // Input dimension has changed!
        output = nullptr;
      }
    }
    if (output == nullptr) {
      output.reset(new Tensor<float>(in->dim(), in->size()));
    }
  }

  void Threshold::forwardProp(std::shared_ptr<TorchData> input) { 
    init(input);
    std::string kernel = jtorch::jtorch_path + "kernels/threshold.cl";
    cl_context->useKernel(kernel.c_str(), "Threshold1D");
    cl_context->setArg(0, TO_TENSOR_PTR(input.get())->storage());
    cl_context->setArg(1, TO_TENSOR_PTR(output.get())->storage());
    cl_context->setArg(2, threshold);
    cl_context->setArg(3, val);
    uint32_t dim = 1;
    uint32_t nelem = TO_TENSOR_PTR(output.get())->nelems();
    cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);
  }

  std::unique_ptr<TorchStage> Threshold::loadFromFile(std::ifstream& file) {
    std::unique_ptr<Threshold> ret(new Threshold());
    file.read((char*)(&ret->threshold), sizeof(ret->threshold));
    file.read((char*)(&ret->val), sizeof(ret->val));
    return std::unique_ptr<TorchStage>(std::move(ret));
  }

}  // namespace jtorch