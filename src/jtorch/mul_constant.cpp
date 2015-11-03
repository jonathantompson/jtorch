#include "jtorch/mul_constant.h"

#include <cstring>

#include "jtorch/tensor.h"
#include "jtorch/jtorch.h"


using namespace jcl::threading;
using namespace jcl::math;

namespace jtorch {

static const char* kMulConstantKernel =
"    __kernel void MulConstant(const __global float* input, const float scalar_constant, __global float* output) {\n"
"\n"
"      const int index = get_global_id(0);\n"
"\n"
"      output[index] = scalar_constant * input[index];\n"
"    }";


MulConstant::MulConstant(float scalar_constant) : TorchStage() {
  output = nullptr;
  scalar_constant_ = scalar_constant;
}

MulConstant::~MulConstant() {}

void MulConstant::init(std::shared_ptr<TorchData> input) {
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

void MulConstant::forwardProp(std::shared_ptr<TorchData> input) {
  init(input);
  cl_context->useKernelCStr(kMulConstantKernel, "MulConstant");
  cl_context->setArg(0, TO_TENSOR_PTR(input.get())->storage());
  cl_context->setArg(1, scalar_constant_);
  cl_context->setArg(2, TO_TENSOR_PTR(output.get())->storage());
  uint32_t dim = 1;
  uint32_t nelem = TO_TENSOR_PTR(output.get())->nelems();
  cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);
}

std::unique_ptr<TorchStage> MulConstant::loadFromFile(std::ifstream& file) {
  float scalar_constant;
  file.read((char*)(&scalar_constant), sizeof(scalar_constant));
  return std::unique_ptr<TorchStage>(new MulConstant(scalar_constant));
}

}  // namespace jtorch
