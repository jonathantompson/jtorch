#include "jtorch/spatial_up_sampling_nearest.h"
#include "jtorch/tensor.h"
#include "jtorch/jtorch.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"

using namespace jcl::threading;
using namespace jcl;

namespace jtorch {

static const char* kSpatialUpSamplingNearest =
"    __kernel void SpatialUpSamplingNearest("
"      const __global  float* input,  /* 0 */"
"      __global  float* output,       /* 1 */"
"      const int scale) {             /* 2 */"
""
"      const int width_out = get_global_size(0);"
"      const int height_out = get_global_size(1);"
""
"      const int width_in = width_out / scale;"
"      const int height_in = height_out / scale;"
""
"      const int x_out = get_global_id(0);"
"      const int y_out = get_global_id(1);"
"      const int f_out = get_global_id(2);"
""
"      const int x_in = x_out / scale;"
"      const int y_in = y_out / scale;"
"      const int f_in = f_out;"
""
"      const int iout = x_out + width_out * (y_out + height_out * f_out);"
"      const int iin = x_in + width_in * (y_in + height_in * f_in);"
""
"      output[iout] = input[iin];"
"    }"
""
"    __kernel void SpatialUpSamplingNearest2D("
"      const __global  float* input,  /* 0 */"
"      __global  float* output,       /* 1 */"
"      const int scale) {             /* 2 */"
""
"      const int width_out = get_global_size(0);"
"      const int height_out = get_global_size(1);"
""
"      const int width_in = width_out / scale;"
"      const int height_in = height_out / scale;"
""
"      const int x_out = get_global_id(0);"
"      const int y_out = get_global_id(1);"
""
"      const int x_in = x_out / scale;"
"      const int y_in = y_out / scale;"
""
"      const int iout = x_out + width_out * (y_out);"
"      const int iin = x_in + width_in * (y_in);"
""
"      output[iout] = input[iin];"
"    }";

SpatialUpSamplingNearest::SpatialUpSamplingNearest(const int32_t scale)
    : TorchStage() {
  scale_ = scale;
  output = nullptr;
  out_size_ = nullptr;
}

SpatialUpSamplingNearest::~SpatialUpSamplingNearest() {}

void SpatialUpSamplingNearest::init(std::shared_ptr<TorchData> input) {
  assert(input->type() == TorchDataType::TENSOR_DATA);

  Tensor<float>* in = TO_TENSOR_PTR(input.get());
  Tensor<float>* out = TO_TENSOR_PTR(output.get());

  assert(in->dim() >= 2);

  if (output != nullptr && in->dim() != out->dim()) {
    output = nullptr;
  }

  // Check that the inner 2 dimensions differ by a single scale
  if (output != nullptr) {
    if (in->size()[0] * scale_ != out->size()[0] ||
        in->size()[1] * scale_ != out->size()[1]) {
      output = nullptr;
    }
  }

  // Check that the remaining dimensions are the same size
  if (output != nullptr) {
    for (uint32_t i = 2; i < in->dim() && output != nullptr; i++) {
      if (in->size()[i] != out->size()[i]) {
        output = nullptr;
      }
    }
  }

  if (output == nullptr) {
    std::unique_ptr<uint32_t[]> out_size(new uint32_t[in->dim()]);
    memcpy(out_size.get(), in->size(), sizeof(out_size[0]) * in->dim());
    out_size[0] *= scale_;
    out_size[1] *= scale_;

    output.reset(new Tensor<float>(in->dim(), out_size.get()));
  }
}

void SpatialUpSamplingNearest::forwardProp(std::shared_ptr<TorchData> input) {
  init(input);

  Tensor<float>* in = TO_TENSOR_PTR(input.get());
  if (in->dim() == 2) {
    cl_context->useKernelCStr(kSpatialUpSamplingNearest,
                              "SpatialUpSamplingNearest2D");
  } else {
    cl_context->useKernelCStr(kSpatialUpSamplingNearest,
                              "SpatialUpSamplingNearest");
  }
  cl_context->setArg(0, TO_TENSOR_PTR(input.get())->storage());
  cl_context->setArg(1, TO_TENSOR_PTR(output.get())->storage());
  cl_context->setArg(2, (int)scale_);
  cl_context->runKernel(jtorch::deviceid, TO_TENSOR_PTR(output.get())->dim(),
                        TO_TENSOR_PTR(output.get())->size(), false);
}

std::unique_ptr<TorchStage> SpatialUpSamplingNearest::loadFromFile(
    std::ifstream& file) {
  int32_t scale;
  file.read((char*)(&scale), sizeof(scale));

  return std::unique_ptr<TorchStage>(new SpatialUpSamplingNearest(scale));
}

}  // namespace jtorch
