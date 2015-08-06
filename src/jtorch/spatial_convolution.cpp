#include "jtorch/spatial_convolution.h"
#include "jtorch/tensor.h"
#include "jtorch/jtorch.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"

using namespace jcl::threading;
using namespace jcl::math;
using namespace jcl;

namespace jtorch {

static const char* kSpatialConvolutionKernel =
"    __kernel void SpatialConvolution("
"      const __global  float* input,  /* 0 */"
"      __global  float* output,       /* 1 */"
"      const __global float* weights, /* 2 */"
"      const __global float* biases,  /* 3 */"
"      const int input_nfeats,        /* 4 */"
"      const int input_height,        /* 5 */"
"      const int input_width,         /* 6 */"
"      const int filt_height,         /* 7 */"
"      const int filt_width) {        /* 8 */"
""
"      const int width = get_global_size(0);"
"      const int height = get_global_size(1);"
"      const int feats = get_global_size(2);"
""
"      const int x_out = get_global_id(0);"
"      const int y_out = get_global_id(1);"
"      const int f_out = get_global_id(2);"
"      const int xInTopLeft = x_out;"
"      const int yInTopLeft = y_out;"
""
"      /* Initilize the output to the bias */"
"      float sum = biases[f_out];"
""
"      const int filt_size = filt_height * filt_width;"
"      const int filt_size_per_fout = input_nfeats * filt_size;"
"      const int in_size = input_width * input_height;"
"      for (int f = 0; f < input_nfeats; f++) {"
"        /* Get a pointer to the current weight matrix and input feature */"
"        /* THIS COULD BE FASTER --> STRIPE WEIGHTS MATRIX FOR BETTER DATA ACCESS! */"
"        const __global  float* pkernel = &weights[f_out * filt_size_per_fout + f * filt_size];"
"        const __global  float* pinput = &input[f * in_size];"
""
"        /* Perform the convolution on this input feature */"
"        for (int r = 0; r < filt_height; r++) {"
"          const int idxFtmp = r * filt_width;"
"          const int yIn = yInTopLeft + r;"
"          const int idxIntmp = yIn * input_width + xInTopLeft;"
"          for (int c = 0; c < filt_width; c++) {"
"            const int idxF  = idxFtmp  + c;"
"            const int idxIn = idxIntmp + c;"
"            sum += pkernel[idxF] * pinput[idxIn];"
"          }"
"        }"
"      }"
"      const int iout = x_out + width * (y_out + height * f_out);"
"      output[iout] = sum;"
"    }"
""
"    __kernel void SpatialConvolutionPadding("
"      const __global  float* input,   /* 0 */"
"      __global  float* output,        /* 1 */"
"      const __global float* weights,  /* 2 */"
"      const __global float* biases,   /* 3 */"
"      const int input_nfeats,         /* 4 */"
"      const int input_height,         /* 5 */"
"      const int input_width,          /* 6 */"
"      const int filt_height,          /* 7 */"
"      const int filt_width,           /* 8 */"
"      const int padding) {            /* 9 */"
""
"      const int width = get_global_size(0);"
"      const int height = get_global_size(1);"
"      const int feats = get_global_size(2);"
""
"      const int x_out = get_global_id(0);"
"      const int y_out = get_global_id(1);"
"      const int f_out = get_global_id(2);"
"      const int xInTopLeft = x_out;"
"      const int yInTopLeft = y_out;"
""
"      const int pad_left_top = padding;"
""
"      /* Initilize the output to the bias */"
"      float sum = biases[f_out];"
""
"      const int filt_size = filt_height * filt_width;"
"      const int filt_size_per_fout = input_nfeats * filt_size;"
"      const int in_size = input_width * input_height;"
"      for (int f = 0; f < input_nfeats; f++) {"
"        /* Get a pointer to the current weight matrix and input feature */"
"        /* THIS COULD BE FASTER --> STRIPE WEIGHTS MATRIX FOR BETTER DATA ACCESS! */"
"        const __global  float* pkernel = &weights[f_out * filt_size_per_fout + f * filt_size];"
"        const __global  float* pinput = &input[f * in_size];"
""
"        /* Perform the convolution on this input feature */"
"        for (int r = 0; r < filt_height; r++) {"
"          const int idxFtmp = r * filt_width;"
"          const int yIn = yInTopLeft + r - pad_left_top;"
"          const int idxIntmp = yIn * input_width;"
""
"          if (yIn >= 0 && yIn < input_height) {"
"            for (int c = 0; c < filt_width; c++) {"
"              const int idxF  = idxFtmp  + c;"
"              const int xIn = xInTopLeft + c - pad_left_top;"
"              if (xIn >= 0 && xIn < input_width) {"
"                const int idxIn = idxIntmp + xIn;"
"                sum += pkernel[idxF] * pinput[idxIn];"
"              }"
"            }"
"          }"
"        }"
"      }"
"      const int iout = x_out + width * (y_out + height * f_out);"
"      output[iout] = sum;"
"    }";

SpatialConvolution::SpatialConvolution(const uint32_t feats_in,
                                       const uint32_t feats_out,
                                       const uint32_t filt_height,
                                       const uint32_t filt_width,
                                       const uint32_t padding)
    : TorchStage() {
  filt_width_ = filt_width;
  filt_height_ = filt_height;
  feats_in_ = feats_in;
  feats_out_ = feats_out;
  padding_ = padding;

  output = nullptr;

  uint32_t dim = 4;
  uint32_t size[4] = {filt_width_, filt_height_, feats_in_, feats_out_};
  weights_.reset(new Tensor<float>(dim, size));
  biases_.reset(new Tensor<float>(1, &feats_out_));
}

SpatialConvolution::~SpatialConvolution() {}

void SpatialConvolution::setWeights(const float* weights) {
  weights_->setData(weights);
}

void SpatialConvolution::setBiases(const float* biases) {
  biases_->setData(biases);
}

void SpatialConvolution::init(std::shared_ptr<TorchData> input) {
  assert(input->type() == TorchDataType::TENSOR_DATA);

  Tensor<float>* in = TO_TENSOR_PTR(input.get());
  assert(in->dim() == 3);
  assert(in->size()[2] == feats_in_);

  if (output != nullptr) {
    uint32_t owidth = in->size()[0] - filt_width_ + 1 + 2 * padding_;
    uint32_t oheight = in->size()[1] - filt_height_ + 1 + 2 * padding_;
    const uint32_t* out_size = TO_TENSOR_PTR(output.get())->size();
    if (out_size[0] != owidth || out_size[1] != oheight ||
        out_size[2] != feats_out_) {
      output = nullptr;
    }
  }
  if (output == nullptr) {
    uint32_t out_dim[3];
    out_dim[0] = in->size()[0] - filt_width_ + 1 + 2 * padding_;
    out_dim[1] = in->size()[1] - filt_height_ + 1 + 2 * padding_;
    out_dim[2] = feats_out_;
    output.reset(new Tensor<float>(3, out_dim));
  }
}

void SpatialConvolution::forwardProp(std::shared_ptr<TorchData> input) {
  init(input);
  Tensor<float>* in = TO_TENSOR_PTR(input.get());
  if (padding_ > 0) {
    cl_context->useKernelCStr(kSpatialConvolutionKernel, "SpatialConvolutionPadding");
  } else {
    cl_context->useKernelCStr(kSpatialConvolutionKernel, "SpatialConvolution");
  }
  cl_context->setArg(0, in->storage());
  cl_context->setArg(1, TO_TENSOR_PTR(output.get())->storage());
  cl_context->setArg(2, weights_->storage());
  cl_context->setArg(3, biases_->storage());
  cl_context->setArg(4, (int)in->size()[2]);
  cl_context->setArg(5, (int)in->size()[1]);
  cl_context->setArg(6, (int)in->size()[0]);
  cl_context->setArg(7, (int)filt_height_);
  cl_context->setArg(8, (int)filt_width_);
  if (padding_ > 0) {
    cl_context->setArg(9, (int)padding_);
  }
  uint32_t dim = 3;
  cl_context->runKernel(jtorch::deviceid, dim,
                        TO_TENSOR_PTR(output.get())->size(), false);
}

std::unique_ptr<TorchStage> SpatialConvolution::loadFromFile(
    std::ifstream& file) {
  int32_t filt_width, filt_height, n_input_features, n_output_features, padding;
  file.read((char*)(&filt_width), sizeof(filt_width));
  file.read((char*)(&filt_height), sizeof(filt_height));
  file.read((char*)(&n_input_features), sizeof(n_input_features));
  file.read((char*)(&n_output_features), sizeof(n_output_features));
  file.read((char*)(&padding), sizeof(padding));

#if defined(DEBUG) || defined(_DEBUG)
  std::cout << "\t\t(fout,fin,kh,kw,pad)=(" << n_output_features << ","
            << n_input_features << "," << filt_height << "," << filt_width
            << "," << padding << ")" << std::endl;
#endif

  std::unique_ptr<SpatialConvolution> ret(new SpatialConvolution(
      n_input_features, n_output_features, filt_height, filt_width, padding));

  int32_t filt_dim = filt_width * filt_height;
  std::unique_ptr<float[]> weights(
      new float[n_output_features * n_input_features * filt_dim]);
  for (int32_t i = 0; i < n_output_features * n_input_features; i++) {
    float* bank = &weights[i * filt_dim];
    file.read((char*)(bank), sizeof(bank[0]) * filt_dim);
  }
  ret->setWeights(weights.get());

  std::unique_ptr<float[]> biases(new float[n_output_features]);
  file.read((char*)(biases.get()), sizeof(biases[0]) * n_output_features);
  ret->setBiases(biases.get());

  return std::unique_ptr<TorchStage>(std::move(ret));
}

}  // namespace jtorch
