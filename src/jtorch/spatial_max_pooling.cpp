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
"                                    const int kw,                  /* 4 */"
"                                    const int kh,                  /* 5 */"
"                                    const int dw,                  /* 6 */"
"                                    const int dh,                  /* 7 */"
"                                    const int padw,                /* 8 */"
"                                    const int padh) {              /* 9 */"
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
"      const int vstart = y_out * dh - padh;"
"      const int vend = vstart + kh - 1;  /* inclusive */"
"      const int ustart = x_out * dw - padw;"
"      const int uend = ustart + kw - 1;  /* inclusive */"
""
"      /* Get a pointer to the current input feature (that corresponds to this */"
"      /* output feature; */"
"      const __global  float* input_f = &input[f_out * input_width * input_height];"
""
"      for (int v = vstart; v <= vend; v++) {"
"        if (v >= 0 && v < input_height) {"
"          for (int u = ustart; u <= uend; u++) {"
"            if (u >= 0 && u < input_width) {"
"              out_val = max(out_val, input_f[v * input_width + u]);"
"            }"
"          }"
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
"                                      const int kw,                  /* 4 */"
"                                      const int kh,                  /* 5 */"
"                                      const int dw,                  /* 6 */"
"                                      const int dh,                  /* 7 */"
"                                      const int padw,                /* 8 */"
"                                      const int padh) {              /* 9 */"
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
"      const int vstart = y_out * dh - padh;"
"      const int vend = vstart + kh - 1;  /* inclusive */"
"      const int ustart = x_out * dw - padw;"
"      const int uend = ustart + kw - 1;  /* inclusive */"
""
"      /* Get a pointer to the current input feature (that corresponds to this */"
"      /* output feature; */"
"      const __global  float* input_f = input;"
""
"      for (int v = vstart; v <= vend; v++) {"
"        if (v >= 0 && v < input_height) {"
"          for (int u = ustart; u <= uend; u++) {"
"            if (u >= 0 && u < input_width) {"
"              out_val = max(out_val, input_f[v * input_width + u]);"
"            }"
"          }"
"        }"
"      }"
""
"      const int index = x_out + width * y_out;"
"      output[index] = out_val;"
"    }";


    SpatialMaxPooling::SpatialMaxPooling(const uint32_t kw, const uint32_t kh, 
      const uint32_t dw, const uint32_t dh, const uint32_t padw,
      const uint32_t padh)
    : TorchStage() {
  kw_ = kw;
  kh_ = kh;
  dw_ = dw;
  dh_ = dh;
  padw_ = padw;
  padh_ = padh;
  output = nullptr;
}

SpatialMaxPooling::~SpatialMaxPooling() {}

void SpatialMaxPooling::init(std::shared_ptr<TorchData> input) {
  RASSERT(input->type() == TorchDataType::TENSOR_DATA);
  Tensor<float>* in = TO_TENSOR_PTR(input.get());
  RASSERT(in->dim() == 2 || in->dim() == 3);

  // We'll escentially do ceil_mode = false from torch
  const uint32_t iwidth = in->size()[0];
  const uint32_t iheight = in->size()[1];
  uint32_t oheight = (long)(floor((float)(iheight - kh_ + 2*padh_) / dh_)) + 1;
  uint32_t owidth  = (long)(floor((float)(iwidth  - kw_ + 2*padw_) / dw_)) + 1;

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
    if (TO_TENSOR_PTR(output.get())->size()[0] != owidth ||
        TO_TENSOR_PTR(output.get())->size()[1] != oheight) {
      output = nullptr;
    }
  }

  if (output == nullptr) {
    std::unique_ptr<uint32_t[]> out_size(new uint32_t[in->dim()]);
    out_size[0] = owidth;
    out_size[1] = oheight;
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
  cl_context->setArg(4, (int)kw_);
  cl_context->setArg(5, (int)kh_);
  cl_context->setArg(6, (int)dw_);
  cl_context->setArg(7, (int)dh_);
  cl_context->setArg(8, (int)padw_);
  cl_context->setArg(9, (int)padh_);
  cl_context->runKernel(jtorch::deviceid, TO_TENSOR_PTR(output.get())->dim(),
                        TO_TENSOR_PTR(output.get())->size(), false);
}

std::unique_ptr<TorchStage> SpatialMaxPooling::loadFromFile(
    std::ifstream& file) {
  int kw, kh, dw, dh, padw, padh;
  file.read((char*)(&kw), sizeof(kw));
  file.read((char*)(&kh), sizeof(kh));
  file.read((char*)(&dw), sizeof(dw));
  file.read((char*)(&dh), sizeof(dh));
  file.read((char*)(&padw), sizeof(padw));
  file.read((char*)(&padh), sizeof(padh));
  return std::unique_ptr<TorchStage>(
    new SpatialMaxPooling(kw, kh, dw, dh, padw, padh));
}

}  // namespace jtorch
