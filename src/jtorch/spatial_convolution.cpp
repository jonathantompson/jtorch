#include "jtorch/spatial_convolution.h"
#include "jtorch/tensor.h"
#include "jtorch/jtorch.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"
#include "jcl/data_str/vector_managed.h"

#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }

using namespace jcl::threading;
using namespace jcl::math;
using namespace jcl::data_str;
using namespace jcl;

namespace jtorch {

  SpatialConvolution::SpatialConvolution(const uint32_t feats_in, 
    const uint32_t feats_out, const uint32_t filt_height, 
    const uint32_t filt_width, const uint32_t padding) : TorchStage() {
    filt_width_ = filt_width;
    filt_height_ = filt_height;
    feats_in_ = feats_in;
    feats_out_ = feats_out;
    padding_ = padding;

    output = NULL;

    uint32_t dim = 4;
    uint32_t size[4] = {filt_width_, filt_height_, feats_in_, feats_out_};
    weights_ = new Tensor<float>(dim, size);
    biases_ = new Tensor<float>(1, &feats_out_);
  }

  SpatialConvolution::~SpatialConvolution() {
    SAFE_DELETE(output);
    SAFE_DELETE(weights_);
    SAFE_DELETE(biases_);
  }

  void SpatialConvolution::setWeights(const float* weights) {
    weights_->setData(weights);
  }

  void SpatialConvolution::setBiases(const float* biases) {
    biases_->setData(biases);
  }

  void SpatialConvolution::init(TorchData& input)  {
    if (input.type() != TorchDataType::TENSOR_DATA) {
      throw std::runtime_error("SpatialConvolution::init() - "
        "FloatTensor expected!");
    }
    Tensor<float>& in = (Tensor<float>&)input;
    if (in.dim() != 3) {
      throw std::runtime_error("SpatialConvolution::init() - Input not 3D!");
    }
    if (in.size()[2] != feats_in_) {
      throw std::runtime_error("SpatialConvolution::init() - ERROR: "
        "incorrect number of input features!");
    }
    if (output != NULL) {
      uint32_t owidth = in.size()[0] - filt_width_ + 1 + 2 * padding_;
      uint32_t oheight  = in.size()[1] - filt_height_ + 1 + 2 * padding_;
      const uint32_t* out_size = TO_TENSOR_PTR(output)->size();
      if (out_size[0] != owidth || out_size[1] != oheight || 
        out_size[2] != feats_out_) {
        SAFE_DELETE(output);
      }
    }
    if (output == NULL) {
      uint32_t out_dim[3];
      out_dim[0] = in.size()[0] - filt_width_ + 1 + 2 * padding_;
      out_dim[1] = in.size()[1] - filt_height_ + 1 + 2 * padding_;
      out_dim[2] = feats_out_;
      output = new Tensor<float>(3, out_dim);
    }
  }

  void SpatialConvolution::forwardProp(TorchData& input) { 
    init(input);
    Tensor<float>& in = (Tensor<float>&)input;
    std::string kernel = jtorch::jtorch_path + "kernels/spatial_convolution.cl";
    if (padding_ > 0) {
      cl_context->useKernel(kernel.c_str(), "SpatialConvolutionPadding");
    } else {
      cl_context->useKernel(kernel.c_str(), "SpatialConvolution");
    }
    cl_context->setArg(0, ((Tensor<float>&)input).storage());
    cl_context->setArg(1, TO_TENSOR_PTR(output)->storage());
    cl_context->setArg(2, weights_->storage());
    cl_context->setArg(3, biases_->storage());
    cl_context->setArg(4, (int)in.size()[2]);
    cl_context->setArg(5, (int)in.size()[1]);
    cl_context->setArg(6, (int)in.size()[0]);
    cl_context->setArg(7, (int)filt_height_);
    cl_context->setArg(8, (int)filt_width_);
    if (padding_ > 0) {
      cl_context->setArg(9, (int)padding_);
    }
    uint32_t dim = 3;
    cl_context->runKernel(jtorch::deviceid, dim, 
      TO_TENSOR_PTR(output)->size(), false);
  }

  TorchStage* SpatialConvolution::loadFromFile(std::ifstream& file) {
    int32_t filt_width, filt_height, n_input_features, n_output_features,
      padding;
    file.read((char*)(&filt_width), sizeof(filt_width));
    file.read((char*)(&filt_height), sizeof(filt_height));
    file.read((char*)(&n_input_features), sizeof(n_input_features));
    file.read((char*)(&n_output_features), sizeof(n_output_features));
    file.read((char*)(&padding), sizeof(padding));

    SpatialConvolution* ret = new SpatialConvolution(n_input_features,
      n_output_features, filt_height, filt_width, padding);

    int32_t filt_dim = filt_width * filt_height;
    float* weights = new float[n_output_features * n_input_features * filt_dim];
    for (int32_t i = 0; i < n_output_features * n_input_features; i++) {
      float* bank = &weights[i * filt_dim];
      file.read((char*)(bank), sizeof(bank[0]) * filt_dim);
    }
    ret->setWeights(weights);
    delete[] weights;

    float* biases = new float[n_output_features];
    file.read((char*)(biases), sizeof(biases[0]) * n_output_features);
    ret->setBiases(biases);
    delete[] biases;

    return ret;
  }

}  // namespace jtorch