#include "jtorch/linear.h"
#include "jtorch/tensor.h"
#include "jtil/exceptions/wruntime_error.h"
#include "jtil/threading/thread.h"
#include "jtil/threading/callback.h"
#include "jtil/threading/thread_pool.h"
#include "jtil/data_str/vector_managed.h"
#include "jcl/jcl.h"

#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }

using namespace jtil::threading;
using namespace jtil::math;
using namespace jtil::data_str;

namespace jtorch {

  Linear::Linear(const int32_t n_inputs, const int32_t n_outputs) 
    : TorchStage() {
    n_inputs_ = n_inputs;
    n_outputs_ = n_outputs;

    output = new Tensor<float>(Int3(n_outputs_, 1, 1));

    weights_ = new Tensor<float>(Int3(n_inputs_, n_outputs_, 1));
    biases_ = new Tensor<float>(n_outputs_);
  }

  Linear::~Linear() {
    SAFE_DELETE(output);
    SAFE_DELETE(weights_);
    SAFE_DELETE(biases_);
  }

  void Linear::setWeights(const float* weights) {
    weights_->setData(weights);
  }

  void Linear::setBiases(const float* biases) {
    biases_->setData(biases);
  }

  void Linear::init(TorchData& input)  {
    if (input.type() != TorchDataType::TENSOR_DATA) {
      throw std::wruntime_error("Linear::init() - "
        "FloatTensor expected!");
    }
    Tensor<float>& in = (Tensor<float>&)input;
    if (static_cast<int32_t>(in.dataSize()) != n_inputs_) {
      throw std::wruntime_error("Linear::init() - ERROR: input size mismatch!");
    }
  }

  void Linear::forwardProp(TorchData& input) { 
    init(input);
    Tensor<float>& in = (Tensor<float>&)input;

#ifdef SIMPLE_LINEAR
    std::string kernel = jtorch::jtorch_path + "kernels/linear.cl";
    cl_context->useKernel(kernel.c_str(), "MatVecMultSimple");
    cl_context->setArg(0, weights_->data());
    cl_context->setArg(1, ((Tensor<float>&)input).data());
    cl_context->setArg(2, ((Tensor<float>*)output)->data());
    cl_context->setArg(3, n_outputs_);
    cl_context->setArg(4, n_inputs_);
    cl_context->runKernel1D(jtorch::deviceid, output->dataSize(), false);
#else
    std::string kernel = jtorch::jtorch_path + "kernels/linear.cl";
    cl_context->useKernel(kernel.c_str(), "MatVecMultThreads");

    int32_t max_worksize =
      cl_context->queryMaxWorkgroupSizeForCurKernel(jtorch::deviceid);
    // http://www.bealto.com/gpu-gemv_v2.html
    // Try and find a good local workgroup size allocation (that is legal)
    // TODO: This is a mess.  Clean it up.
    Int3 max_item_size;
    cl_context->getMaxWorkitemSizes(jtorch::deviceid, max_item_size);
    uint32_t p = std::min<int32_t>(16, std::min<int32_t>(max_item_size[1],
      max_worksize));
    Int2 global_size(n_outputs_, p);
    Int2 local_size(std::min<int>(n_outputs_ / p + 1, 
      cl_context->getMaxWorkgroupSize(jtorch::deviceid) / p), p);  // Maximum
    while ((n_outputs_ % local_size[0] != 0 ||
      local_size[0] * local_size[1] > max_worksize) && local_size[0] > 1) {
      local_size[0]--;
    }

    cl_context->setArg(0, weights_->data());
    cl_context->setArg(1, in.data());
    cl_context->setArg(2, ((Tensor<float>*)output)->data());
    float dummy; static_cast<void>(dummy);
    // setArg with NULL --> Local memory allocation (per local workgroup)
    cl_context->setArg(3, sizeof(dummy) * local_size[0] * local_size[1], NULL);
    cl_context->setArg(4, n_outputs_);
    cl_context->setArg(5, n_inputs_);

    cl_context->runKernel2D(jtorch::deviceid, global_size, local_size, false);
#endif

    // Now add in the bias
    cl_context->useKernel(kernel.c_str(), "Accum");
    cl_context->setArg(0, ((Tensor<float>*)output)->data());
    cl_context->setArg(1, biases_->data());
    cl_context->runKernel1D(jtorch::deviceid, output->dataSize(), false);
  }

  TorchStage* Linear::loadFromFile(std::ifstream& file) {
    int32_t n_outputs;
    int32_t n_inputs;
    file.read((char*)(&n_outputs), sizeof(n_outputs));
    file.read((char*)(&n_inputs), sizeof(n_inputs));
    Linear* ret = new Linear(n_inputs, n_outputs);

    int32_t n_weights = n_outputs * n_inputs;
    float* weights_cpu = new float[n_weights];
    file.read((char*)(weights_cpu), sizeof(weights_cpu[0]) * n_weights);
    ret->setWeights(weights_cpu);
    delete[] weights_cpu;

    float* bias_cpu = new float[n_outputs];
    file.read((char*)(bias_cpu), sizeof(bias_cpu[0]) * n_outputs);
    ret->setBiases(bias_cpu);
    delete[] bias_cpu;

    return ret;
  }

}  // namespace jtorch
