#include "jtorch/tanh.h"
#include "jtorch/jtorch.h"
#include "jtorch/tensor.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"

using namespace jcl::threading;
using namespace jcl::math;

namespace jtorch {

static const char* kTanhKernel =
"    __kernel void TanH(const __global float* input, __global float* output) {\n"
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
"      output[index] = tanh(input[index]);\n"
"    }\n"
"\n"
"    __kernel void TanH1D(const __global float* input, __global float* output) {\n"
"\n"
"      const int x_out = get_global_id(0);\n"
"\n"
"      output[x_out] = tanh(input[x_out]);\n"
"    }";


Tanh::Tanh() : TorchStage() { output = nullptr; }

Tanh::~Tanh() {}

void Tanh::init(std::shared_ptr<TorchData> input) {
  RASSERT(input->type() == TorchDataType::TENSOR_DATA);
  Tensor<float>* in = TO_TENSOR_PTR(input.get());
  Tensor<float>* out = TO_TENSOR_PTR(output.get());
  if (output != nullptr) {
    if (!out->isSameSizeAs(*in)) {
      // Input dimension has changed!
      output = nullptr;
    }
  }
  if (output == nullptr) {
    output.reset(new Tensor<float>(in->dim(), in->size()));
  }
}

void Tanh::forwardProp(std::shared_ptr<TorchData> input) {
  init(input);
  cl_context->useKernelCStr(kTanhKernel, "TanH1D");
  cl_context->setArg(0, TO_TENSOR_PTR(input.get())->storage());
  cl_context->setArg(1, TO_TENSOR_PTR(output.get())->storage());
  uint32_t dim = 1;
  uint32_t nelem = TO_TENSOR_PTR(output.get())->nelems();
  cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);
}

std::unique_ptr<TorchStage> Tanh::loadFromFile(std::ifstream& file) {
  // Nothing to do for Tanh
  return std::unique_ptr<TorchStage>(new Tanh());
}

}  // namespace jtorch
