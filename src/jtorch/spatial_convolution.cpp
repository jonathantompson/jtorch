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

  SpatialConvolution::SpatialConvolution(const int32_t feats_in, 
    const int32_t feats_out, const int32_t filt_height, 
    const int32_t filt_width) 
    : TorchStage() {
    filt_width_ = filt_width;
    filt_height_ = filt_height;
    feats_in_ = feats_in;
    feats_out_ = feats_out;

    output = NULL;

    weights_ = new Tensor<float>(Int3(filt_width_, filt_height_,
      feats_out_ * feats_in_));
    biases_ = new Tensor<float>(feats_out_);
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
    if (in.dim()[2] != feats_in_) {
      throw std::runtime_error("SpatialConvolution::init() - ERROR: "
        "incorrect number of input features!");
    }
    if (output != NULL) {
      Int3 out_dim(in.dim());
      out_dim[0] = out_dim[0] - filt_width_ + 1;
      out_dim[1] = out_dim[1] - filt_height_ + 1;
      out_dim[2] = feats_out_;
      if (!Int3::equal(out_dim, ((Tensor<float>*)output)->dim())) {
        // Input dimension has changed!
        SAFE_DELETE(output);
      }
    }
    if (output == NULL) {
      Int3 out_dim(in.dim());
      out_dim[0] = out_dim[0] - filt_width_ + 1;
      out_dim[1] = out_dim[1] - filt_height_ + 1;
      out_dim[2] = feats_out_;
      output = new Tensor<float>(out_dim);
      //cl_context->getOptimalLocalWorkgroupSizes(deviceid, 
      //  ((Tensor<float>*)output)->dim(), local_worgroup_size);
    }
  }

  void SpatialConvolution::forwardProp(TorchData& input) { 
    init(input);
    Tensor<float>& in = (Tensor<float>&)input;
    std::string kernel = jtorch::jtorch_path + "kernels/spatial_convolution.cl";
    cl_context->useKernel(kernel.c_str(), "SpatialConvolution");
    cl_context->setArg(0, ((Tensor<float>&)input).data());
    cl_context->setArg(1, ((Tensor<float>*)output)->data());
    cl_context->setArg(2, weights_->data());
    cl_context->setArg(3, biases_->data());
    cl_context->setArg(4, (int)in.dim()[2]);
    cl_context->setArg(5, (int)in.dim()[1]);
    cl_context->setArg(6, (int)in.dim()[0]);
    cl_context->setArg(7, (int)filt_height_);
    cl_context->setArg(8, (int)filt_width_);
    cl_context->runKernel3D(jtorch::deviceid, ((Tensor<float>*)output)->dim(),
      false);
  }

  TorchStage* SpatialConvolution::loadFromFile(std::ifstream& file) {
    int32_t filt_width, filt_height, n_input_features, n_output_features;
    file.read((char*)(&filt_width), sizeof(filt_width));
    file.read((char*)(&filt_height), sizeof(filt_height));
    file.read((char*)(&n_input_features), sizeof(n_input_features));
    file.read((char*)(&n_output_features), sizeof(n_output_features));

    SpatialConvolution* ret = new SpatialConvolution(n_input_features,
      n_output_features, filt_height, filt_width);

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