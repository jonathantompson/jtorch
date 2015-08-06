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
  std::string kernel = jtorch::jtorch_path + "kernels/spatial_convolution.cl";
  if (padding_ > 0) {
    cl_context->useKernel(kernel.c_str(), "SpatialConvolutionPadding");
  } else {
    cl_context->useKernel(kernel.c_str(), "SpatialConvolution");
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
