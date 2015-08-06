#include "jtorch/spatial_subtractive_normalization.h"
#include "jtorch/tensor.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"

using namespace jcl::threading;
using namespace jcl::math;

namespace jtorch {

static const char* kSpatialSubtractiveNormalizationKernel =
"    __kernel void SpatialSubtractiveNormalizationHoriz("
"      const __global float* input,     /* 0 */"
"      __global float* output,          /* 1 */"
"      const __global float* kernel1d,  /* 2 */"
"      const int filt_rad) {            /* 3 */"
""
"      const int width = get_global_size(0);"
"      const int height = get_global_size(1);"
""
"      const int x_out = get_global_id(0);"
"      const int y_out = get_global_id(1);"
"      const int f_out = get_global_id(2);"
""
"      /* Initilize the output to zero and accumulate the input values */"
"      float sum = 0;"
""
"      const int iout = x_out + width * (y_out + height * f_out);"
""
"      int i = 0;"
"      for (int u_offset = -filt_rad; u_offset <= filt_rad; u_offset++, i++) {"
"        int u = x_out + u_offset;"
"        if (u >= 0 && u < width) {"
"          sum += kernel1d[i] * input[iout + u_offset];"
"        }"
"      }"
""
"      output[iout] = sum;"
"    }"
""
"    __kernel void SpatialSubtractiveNormalizationVert("
"      const __global float* input,     /* 0 */"
"      __global  float* output,         /* 1 */"
"      const __global float* kernel1d,  /* 2 */"
"      const int filt_rad) {            /* 3 */"
""
"      const int width = get_global_size(0);"
"      const int height = get_global_size(1);"
""
"      const int x_out = get_global_id(0);"
"      const int y_out = get_global_id(1);"
"      const int f_out = get_global_id(2);"
""
"      /* Initilize the output to zero and accumulate the input values */"
"      float sum = 0;"
""
"      const int iout = x_out + width * (y_out + height * f_out);"
""
"      int i = 0;"
"      for (int v_offset = -filt_rad; v_offset <= filt_rad; v_offset++, i++) {"
"        int v = y_out + v_offset;"
"        if (v >= 0 && v < height) {"
"          sum += kernel1d[i] * input[iout + v_offset * width];"
"        }"
"      }"
""
"      output[iout] = sum;"
"    }"
""
"    __kernel void SpatialSubtractiveNormalization2D("
"      const __global float* input,     /* 0 */"
"      __global  float* output,         /* 1 */"
"      const __global float* kernel2d,  /* 2 */"
"      const int filt_rad_u,            /* 3 */"
"      const int filt_rad_v) {          /* 4 */"
""
"      const int width = get_global_size(0);"
"      const int height = get_global_size(1);"
""
"      const int x_out = get_global_id(0);"
"      const int y_out = get_global_id(1);"
"      const int f_out = get_global_id(2);"
""
"      /* Initilize the output to zero and accumulate the input values */"
"      float sum = 0;"
""
"      const int iout = x_out + width * (y_out + height * f_out);"
"      const int filt_size_u = 2 * filt_rad_u + 1;"
""
"      for (int v_offset = -filt_rad_v; v_offset <= filt_rad_v; v_offset++) {"
"        int v = y_out + v_offset;"
"        int v_filt = v_offset + filt_rad_v;"
"        for (int u_offset = -filt_rad_u; u_offset <= filt_rad_u; u_offset++) {"
"          int u = x_out + u_offset;"
"          int u_filt = u_offset + filt_rad_u;"
"          if (v >= 0 && v < height && u >= 0 && u < width) {"
"            sum += kernel2d[v_filt * filt_size_u + u_filt] * "
"              input[iout + v_offset * width + u_offset];"
"          }"
"        }"
"      }"
""
"      output[iout] = sum;"
"    }"
""
"    __kernel void SpatialSubtractiveNormalizationAccumDiv("
"      const __global float* input,       /* 0 */"
"      __global  float* output,           /* 1 */"
"      const __global float* mean_coeff,  /* 2 */"
"      const int input_nfeats) {          /* 3 */"
""
"      const int width = get_global_size(0);"
"      const int height = get_global_size(1);"
""
"      const int x_out = get_global_id(0);"
"      const int y_out = get_global_id(1);"
"      const int f_out = get_global_id(2);"
""
"      /* Initilize the output to zero and accumulate the input values */"
"      float sum = 0;"
""
"      const int uvout = x_out + width * y_out;  /* index on each input image */"
"      const int im_dim = width * height;"
"      for (int f = 0; f < input_nfeats; f++) {"
"        sum += input[f * im_dim + uvout];"
"      }"
""
"      output[uvout] = sum / ((float)input_nfeats * (float)input_nfeats * mean_coeff[uvout]);"
"    }"
""
"    __kernel void SpatialSubtractiveNormalization("
"      const __global float* input,      /* 0 */"
"      __global float* output,           /* 1 */"
"      const __global float* mean) {     /* 2 */"
""
"      const int width = get_global_size(0);"
"      const int height = get_global_size(1);"
""
"      const int x_out = get_global_id(0);"
"      const int y_out = get_global_id(1);"
"      const int f_out = get_global_id(2);"
""
"      const int index = x_out + width * (y_out + height * f_out);"
"      output[index] = input[index] - mean[y_out * width + x_out];"
"    }";


// kernel1d default is either TorchStage::gaussian1D<float>(n) or just a
// vector of 1 values.
SpatialSubtractiveNormalization::SpatialSubtractiveNormalization(
    const std::shared_ptr<Tensor<float>> kernel)
    : TorchStage() {
  assert(kernel->dim() <= 2);

  // Averaging kernel must have odd size!
  assert(kernel->size()[0] % 2 != 0 &&
         !(kernel->dim() == 2 && kernel->size()[1] % 2 == 0));

  // Clone and normalize the input kernel
  kernel_.reset(Tensor<float>::clone(*kernel.get()));
  float sum = Tensor<float>::slowSum(*kernel_);
  Tensor<float>::div(*kernel_, sum);

  output = nullptr;
  mean_coef_.reset(nullptr);
  mean_pass1_.reset(nullptr);
  mean_pass2_.reset(nullptr);
  mean_.reset(nullptr);
}

SpatialSubtractiveNormalization::~SpatialSubtractiveNormalization() {
  cleanup();
}

void SpatialSubtractiveNormalization::cleanup() {
  output = nullptr;
  mean_coef_.reset(nullptr);
  mean_pass1_.reset(nullptr);
  mean_pass2_.reset(nullptr);
  mean_.reset(nullptr);
}

void SpatialSubtractiveNormalization::init(std::shared_ptr<TorchData> input) {
  assert(input->type() == TorchDataType::TENSOR_DATA);
  Tensor<float>* in = TO_TENSOR_PTR(input.get());

  assert(in->dim() == 3);

  if (output != nullptr) {
    if (!in->isSameSizeAs(*TO_TENSOR_PTR(output.get()))) {
      // Input dimension has changed!
      cleanup();
    }
  }

  if (output == nullptr) {
    output.reset(new Tensor<float>(in->dim(), in->size()));
    mean_pass1_.reset(new Tensor<float>(in->dim(), in->size()));
    mean_pass2_.reset(new Tensor<float>(in->dim(), in->size()));
  }

  if (mean_coef_ == nullptr) {
    uint32_t mean_coeff_size[2];
    mean_coeff_size[0] = TO_TENSOR_PTR(output.get())->size()[0];
    mean_coeff_size[1] = TO_TENSOR_PTR(output.get())->size()[1];
    mean_coef_.reset(new Tensor<float>(2, mean_coeff_size));

    std::unique_ptr<float[]> mean_coef_cpu(new float[mean_coef_->nelems()]);
    std::unique_ptr<float[]> kernel_cpu(new float[kernel_->nelems()]);
    kernel_->getData(kernel_cpu.get());
    bool onedim_kernel = kernel_->dim() == 1;

    // Filter an image of all 1 values to create the normalization constants
    // See norm_test.lua for proof that this works as well as:
    // https://github.com/andresy/torch/blob/master/extra/nn/SpatialSubtractiveNormalization.lua
    int32_t n_feats = TO_TENSOR_PTR(output.get())->size()[2];
    int32_t height = TO_TENSOR_PTR(output.get())->size()[1];
    int32_t width = TO_TENSOR_PTR(output.get())->size()[0];
    if (onedim_kernel) {
      // 1D case - The filter is seperable, but we'll just do the dumb 2D
      // version since we only do this once on startup.  --> O(n * m)
      uint32_t kernel_size = kernel_->size()[0];
      int32_t filt_rad = (kernel_size - 1) / 2;
      for (int32_t v = 0; v < height; v++) {
        for (int32_t u = 0; u < width; u++) {
          float tmp = 0.0f;
          for (int32_t v_filt = -filt_rad; v_filt <= filt_rad; v_filt++) {
            for (int32_t u_filt = -filt_rad; u_filt <= filt_rad; u_filt++) {
              int32_t u_in = u + u_filt;
              int32_t v_in = v + v_filt;
              if (u_in >= 0 && u_in < width && v_in >= 0 && v_in < height) {
                // Pixel is inside --> We'll effectively clamp zeros elsewhere.
                tmp += (kernel_cpu[v_filt + filt_rad] *
                        kernel_cpu[u_filt + filt_rad]);
              }
            }
          }
          mean_coef_cpu[v * width + u] = tmp / n_feats;
        }
      }
    } else {
      // 2D case
      int32_t kernel_size_u = kernel_->size()[0];
      int32_t kernel_size_v = kernel_->size()[1];
      int32_t filt_rad_u = (kernel_size_u - 1) / 2;
      int32_t filt_rad_v = (kernel_size_v - 1) / 2;
      for (int32_t v = 0; v < height; v++) {
        for (int32_t u = 0; u < width; u++) {
          float tmp = 0.0f;
          for (int32_t v_filt = -filt_rad_v; v_filt <= filt_rad_v; v_filt++) {
            for (int32_t u_filt = -filt_rad_u; u_filt <= filt_rad_u; u_filt++) {
              int32_t u_in = u + u_filt;
              int32_t v_in = v + v_filt;
              if (u_in >= 0 && u_in < width && v_in >= 0 && v_in < height) {
                // Pixel is inside --> We'll effectively clamp zeros elsewhere.
                tmp += kernel_cpu[(v_filt + filt_rad_v) * kernel_size_u +
                                  (u_filt + filt_rad_u)];
              }
            }
          }
          mean_coef_cpu[v * width + u] = tmp / n_feats;
        }
      }
    }
    mean_coef_->setData(mean_coef_cpu.get());
  }
  if (mean_ == nullptr) {
    uint32_t mean_coeff_size[2];
    mean_coeff_size[0] = TO_TENSOR_PTR(output.get())->size()[0];
    mean_coeff_size[1] = TO_TENSOR_PTR(output.get())->size()[1];
    mean_.reset(new Tensor<float>(2, mean_coeff_size));
  }
}

void SpatialSubtractiveNormalization::forwardProp(
    std::shared_ptr<TorchData> input) {
  init(input);
  bool onedim_kernel = kernel_->dim() == 1;

  Tensor<float>* in = TO_TENSOR_PTR(input.get());
  Tensor<float>* out = TO_TENSOR_PTR(output.get());

  if (onedim_kernel) {
    int32_t filt_rad = ((int32_t)kernel_->size()[0] - 1) / 2;

    // Perform horizontal filter pass
    cl_context->useKernelCStr(kSpatialSubtractiveNormalizationKernel,
                              "SpatialSubtractiveNormalizationHoriz");
    cl_context->setArg(0, in->storage());
    cl_context->setArg(1, mean_pass1_->storage());
    cl_context->setArg(2, kernel_->storage());
    cl_context->setArg(3, filt_rad);
    cl_context->runKernel(jtorch::deviceid, mean_pass1_->dim(),
                          mean_pass1_->size(), false);

    // Perform vertical filter pass
    cl_context->useKernelCStr(kSpatialSubtractiveNormalizationKernel,
                              "SpatialSubtractiveNormalizationVert");
    cl_context->setArg(0, mean_pass1_->storage());
    cl_context->setArg(1, mean_pass2_->storage());
    cl_context->setArg(2, kernel_->storage());
    cl_context->setArg(3, filt_rad);
    cl_context->runKernel(jtorch::deviceid, mean_pass2_->dim(),
                          mean_pass2_->size(), false);
  } else {
    int32_t filt_rad_u = ((int32_t)kernel_->size()[0] - 1) / 2;
    int32_t filt_rad_v = ((int32_t)kernel_->size()[1] - 1) / 2;

    // Perform horizontal filter pass
    cl_context->useKernelCStr(kSpatialSubtractiveNormalizationKernel,
                              "SpatialSubtractiveNormalization2D");
    cl_context->setArg(0, in->storage());
    cl_context->setArg(1, mean_pass2_->storage());
    cl_context->setArg(2, kernel_->storage());
    cl_context->setArg(3, filt_rad_u);
    cl_context->setArg(4, filt_rad_v);
    cl_context->runKernel(jtorch::deviceid, mean_pass2_->dim(),
                          mean_pass2_->size(), false);
  }

  // Perform accumulation and division pass
  cl_context->useKernelCStr(kSpatialSubtractiveNormalizationKernel,
                            "SpatialSubtractiveNormalizationAccumDiv");
  cl_context->setArg(0, mean_pass2_->storage());
  cl_context->setArg(1, mean_->storage());
  cl_context->setArg(2, mean_coef_->storage());
  cl_context->setArg(3, (int)out->size()[2]);
  cl_context->runKernel(jtorch::deviceid, mean_->dim(), mean_->size(), false);

  // Perform normalization pass
  cl_context->useKernelCStr(kSpatialSubtractiveNormalizationKernel,
                            "SpatialSubtractiveNormalization");
  cl_context->setArg(0, in->storage());
  cl_context->setArg(1, out->storage());
  cl_context->setArg(2, mean_->storage());
  cl_context->runKernel(jtorch::deviceid, out->dim(), out->size(), false);
}

std::unique_ptr<TorchStage> SpatialSubtractiveNormalization::loadFromFile(
    std::ifstream& file) {
  // This whole thing is a little wasteful.  I copy to GPU here, and then
  // I copy it back down in the constructor anyway...  But it's good enough
  // for now.
  int32_t kernel_size_2, kernel_size_1;  // kernel_size_1 is the inner dim
  file.read((char*)(&kernel_size_1), sizeof(kernel_size_1));
  file.read((char*)(&kernel_size_2), sizeof(kernel_size_2));
  std::shared_ptr<Tensor<float>> kernel = nullptr;
  if (kernel_size_2 > 1) {
    // The kernel is 2D
    uint32_t dim = 2;
    uint32_t size[2] = {static_cast<uint32_t>(kernel_size_1),
                        static_cast<uint32_t>(kernel_size_2)};
    kernel.reset(new Tensor<float>(dim, size));
  } else {
    uint32_t dim = 1;
    uint32_t size[1] = {static_cast<uint32_t>(kernel_size_1)};
    kernel.reset(new Tensor<float>(dim, size));
  }
  std::unique_ptr<float[]> kernel_cpu(new float[kernel->nelems()]);
  file.read((char*)(kernel_cpu.get()),
            kernel->nelems() * sizeof(kernel_cpu[0]));
  kernel->setData(kernel_cpu.get());
  return std::unique_ptr<TorchStage>(
      new SpatialSubtractiveNormalization(kernel));
}

}  // namespace jtorch
