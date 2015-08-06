#include "jtorch/spatial_max_pooling.h"
#include "jtorch/tensor.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"

using namespace jcl::threading;
using namespace jcl::math;

namespace jtorch {

static const char* kSpatialMaxPoolingKernel =
"    __kernel void SpatialMaxPooling(const __global  float* input,  /* 0 */"
"                                    __global  float* output,       /* 1 */"
"                                    const int input_height,        /* 2 */"
"                                    const int input_width,         /* 3 */"
"                                    const int poolsize_v,          /* 4 */"
"                                    const int poolsize_u) {        /* 5 */"
""
"      const int width = get_global_size(0);"
"      const int height = get_global_size(1);"
"      const int feats = get_global_size(2);"
""
"      const int x_out = get_global_id(0);"
"      const int y_out = get_global_id(1);"
"      const int f_out = get_global_id(2);"
""
"      /* Initilize the output to the bias */"
"      float out_val = - INFINITY;"
""
"      const int vstart = y_out * poolsize_v;"
"      const int vend = (y_out + 1) * poolsize_v - 1;"
""
"      /* Get a pointer to the current input feature (that corresponds to this */"
"      /* output feature; */"
"      const __global  float* input_f = &input[f_out * input_width * input_height];"
""
"      for (int v = vstart; v <= vend; v++) {"
"        const int istart = v * input_width + x_out * poolsize_u;"
"        const int iend = v * input_width + (x_out + 1) * poolsize_u - 1;"
""
"        for (int i = istart; i <= iend; i++) {"
"          out_val = max(out_val, input_f[i]);"
"        }"
"      }"
""
"      const int index = x_out + width * (y_out + height * f_out);"
"      output[index] = out_val;"
"    }"
""
"    __kernel void SpatialMaxPooling2D(const __global  float* input,  /* 0 */"
"                                      __global  float* output,       /* 1 */"
"                                      const int input_height,        /* 2 */"
"                                      const int input_width,         /* 3 */"
"                                      const int poolsize_v,          /* 4 */"
"                                      const int poolsize_u) {        /* 5 */"
""
"      const int width = get_global_size(0);"
"      const int height = get_global_size(1);"
""
"      const int x_out = get_global_id(0);"
"      const int y_out = get_global_id(1);"
""
"      /* Initilize the output to the bias */"
"      float out_val = - INFINITY;"
""
"      const int vstart = y_out * poolsize_v;"
"      const int vend = (y_out + 1) * poolsize_v - 1;"
""
"      /* Get a pointer to the current input feature (that corresponds to this */"
"      /* output feature; */"
"      const __global  float* input_f = input;"
""
"      for (int v = vstart; v <= vend; v++) {"
"        const int istart = v * input_width + x_out * poolsize_u;"
"        const int iend = v * input_width + (x_out + 1) * poolsize_u - 1;"
""
"        for (int i = istart; i <= iend; i++) {"
"          out_val = max(out_val, input_f[i]);"
"        }"
"      }"
""
"      const int index = x_out + width * y_out;"
"      output[index] = out_val;"
"    }";


    SpatialMaxPooling::SpatialMaxPooling(const uint32_t poolsize_v,
                                         const uint32_t poolsize_u)
    : TorchStage() {
  poolsize_v_ = poolsize_v;
  poolsize_u_ = poolsize_u;
  output = nullptr;
}

SpatialMaxPooling::~SpatialMaxPooling() {}

void SpatialMaxPooling::init(std::shared_ptr<TorchData> input) {
  assert(input->type() == TorchDataType::TENSOR_DATA);
  Tensor<float>* in = TO_TENSOR_PTR(input.get());
  assert(in->dim() == 2 || in->dim() == 3);

  if (output != nullptr && TO_TENSOR_PTR(output.get())->dim() != in->dim()) {
    // Input dimension has changed!
    output = nullptr;
  }

  if (output != nullptr) {
    // Check that the dimensions above the lowest 2 match
    for (uint32_t i = 2; i < in->dim() && output != nullptr; i++) {
      if (TO_TENSOR_PTR(output.get())->size()[i] != in->size()[i]) {
        output = nullptr;
      }
    }
  }

  if (output != nullptr) {
    // Check that the lowest 2 dimensions are the correct size
    if (TO_TENSOR_PTR(output.get())->size()[0] != in->size()[0] / poolsize_u_ ||
        TO_TENSOR_PTR(output.get())->size()[1] != in->size()[1] / poolsize_v_) {
      output = nullptr;
    }
  }

  if (output == nullptr) {
    // Check that the width and height is a multiple of the poolsize
    assert(in->size()[0] % poolsize_u_ == 0 &&
           in->size()[1] % poolsize_v_ == 0);
    std::unique_ptr<uint32_t[]> out_size(new uint32_t[in->dim()]);
    out_size[0] = in->size()[0] / poolsize_u_;
    out_size[1] = in->size()[1] / poolsize_v_;
    for (uint32_t i = 2; i < in->dim(); i++) {
      out_size[i] = in->size()[i];
    }
    output.reset(new Tensor<float>(in->dim(), out_size.get()));
  }
}

void SpatialMaxPooling::forwardProp(std::shared_ptr<TorchData> input) {
  init(input);
  bool two_dim = TO_TENSOR_PTR(input.get())->dim() == 2;
  if (two_dim) {
    cl_context->useKernelCStr(kSpatialMaxPoolingKernel, "SpatialMaxPooling2D");
  } else {
    cl_context->useKernelCStr(kSpatialMaxPoolingKernel, "SpatialMaxPooling");
  }
  cl_context->setArg(0, TO_TENSOR_PTR(input.get())->storage());
  cl_context->setArg(1, TO_TENSOR_PTR(output.get())->storage());
  cl_context->setArg(2, (int)TO_TENSOR_PTR(input.get())->size()[1]);
  cl_context->setArg(3, (int)TO_TENSOR_PTR(input.get())->size()[0]);
  cl_context->setArg(4, (int)poolsize_v_);
  cl_context->setArg(5, (int)poolsize_u_);
  cl_context->runKernel(jtorch::deviceid, TO_TENSOR_PTR(output.get())->dim(),
                        TO_TENSOR_PTR(output.get())->size(), false);
}

std::unique_ptr<TorchStage> SpatialMaxPooling::loadFromFile(
    std::ifstream& file) {
  int poolu, poolv;
  file.read((char*)(&poolu), sizeof(poolu));
  file.read((char*)(&poolv), sizeof(poolv));
  return std::unique_ptr<TorchStage>(new SpatialMaxPooling(poolu, poolv));
}

}  // namespace jtorch
