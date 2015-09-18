#include "jtorch/threshold.h"
#include "jtorch/jtorch.h"
#include "jtorch/tensor.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"

using namespace jcl::threading;
using namespace jcl::math;

namespace jtorch {

static const char* kThresholdKernel =
"    __kernel void Threshold(\n"
"      const __global  float* input, \n"
"      __global float* output,\n"
"      const float threshold, \n"
"      const float val) {\n"
"\n"
"      const int width = get_global_size(0);\n"
"      const int height = get_global_size(1);\n"
"\n"
"      const int x_out = get_global_id(0);\n"
"      const int y_out = get_global_id(1);\n"
"      const int f_out = get_global_id(2);\n"
"\n"
"      const int index = x_out + width * (y_out + height * f_out);\n"
"\n"
"      output[index] = input[index] > threshold ? input[index] : val;\n"
"    }\n"
"\n"
"    __kernel void Threshold1D(\n"
"      const __global  float* input, \n"
"      __global float* output,\n"
"      const float threshold, \n"
"      const float val) {\n"
"\n"
"      const int x_out = get_global_id(0);\n"
"\n"
"      output[x_out] = input[x_out] > threshold ? input[x_out] : val;\n"
"    }";



Threshold::Threshold() : TorchStage() {
  output = nullptr;
  threshold = 1e-6f;
  val = 0;
}

Threshold::~Threshold() {}

void Threshold::init(std::shared_ptr<TorchData> input) {
  RASSERT(input->type() == TorchDataType::TENSOR_DATA);
  Tensor<float>* in = TO_TENSOR_PTR(input.get());
  if (output != nullptr) {
    if (!in->isSameSizeAs(*TO_TENSOR_PTR(output.get()))) {
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
  cl_context->useKernelCStr(kThresholdKernel, "Threshold1D");
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
