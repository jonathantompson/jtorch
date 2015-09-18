#include "jtorch/spatial_batch_normalization.h"
#include "jtorch/tensor.h"
#include "jtorch/jtorch.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"

using namespace jcl::threading;
using namespace jcl;

namespace jtorch {

static const char* kSpatialBatchNormalizationKernel =
"    __kernel void SpatialBatchNormalizationAffine("
"      const __global float* input,         /* 0 */"
"      const __global float* running_mean,  /* 1 */"
"      const __global float* running_std,   /* 2 */"
"      __global  float* output,             /* 3 */"
"      const __global float* weights,       /* 4 */"
"      const __global float* biases) {      /* 5 */"
""
"      const int width = get_global_size(0);"
"      const int height = get_global_size(1);"
""
"      const int x = get_global_id(0);"
"      const int y = get_global_id(1);"
"      const int f = get_global_id(2);"
""
"      const int i = x + width * (y + height * f);"
""
"      output[i] = (input[i] - running_mean[f]) * running_std[f] * weights[f] +"
"                  biases[f];"
"    };"
""
"    __kernel void SpatialBatchNormalization("
"      const __global float* input,         /* 0 */"
"      const __global float* running_mean,  /* 1 */"
"      const __global float* running_std,   /* 2 */"
"      __global  float* output) {           /* 3 */"
""
"      const int width = get_global_size(0);"
"      const int height = get_global_size(1);"
""
"      const int x = get_global_id(0);"
"      const int y = get_global_id(1);"
"      const int f = get_global_id(2);"
""
"      const int i = x + width * (y + height * f);"
""
"      output[i] = (input[i] - running_mean[f]) * running_std[f];"
"    };";

SpatialBatchNormalization::SpatialBatchNormalization(const bool affine, 
  const uint32_t nfeats) : TorchStage() {
  affine_ = affine;
  nfeats_ = nfeats;
  const uint32_t dim = 1;
  const uint32_t size[dim] = {nfeats};
  weights_.reset(new Tensor<float>(dim, size));
  biases_.reset(new Tensor<float>(dim, size));
  running_mean_.reset(new Tensor<float>(dim, size));
  running_std_.reset(new Tensor<float>(dim, size));
}

SpatialBatchNormalization::~SpatialBatchNormalization() {}

void SpatialBatchNormalization::init(std::shared_ptr<TorchData> input) {
  RASSERT(input->type() == TorchDataType::TENSOR_DATA);

  Tensor<float>* in = TO_TENSOR_PTR(input.get());
  Tensor<float>* out = TO_TENSOR_PTR(output.get());

  RASSERT(in->dim() >= 3);
  RASSERT(in->size()[2] == nfeats_);

  if (output != nullptr && in->dim() != out->dim()) {
    output = nullptr;
  }

  // Check that the input and output size are the same.
  if (output != nullptr) {
    if (in->size()[0] != out->size()[0] ||
        in->size()[1] != out->size()[1] ||
        in->size()[2] != out->size()[2]) {
      output = nullptr;
    }
  }

  if (output == nullptr) {
    output.reset(new Tensor<float>(in->dim(), in->size()));
  }
}

void SpatialBatchNormalization::forwardProp(std::shared_ptr<TorchData> input) {
  init(input);

  Tensor<float>* in = TO_TENSOR_PTR(input.get());
  Tensor<float>* out = TO_TENSOR_PTR(output.get());
  if (affine_) {
    cl_context->useKernelCStr(kSpatialBatchNormalizationKernel,
                              "SpatialBatchNormalizationAffine");
  } else {
    cl_context->useKernelCStr(kSpatialBatchNormalizationKernel,
                              "SpatialBatchNormalization");
  }
  cl_context->setArg(0, in->storage());
  cl_context->setArg(1, TO_TENSOR_PTR(running_mean_.get())->storage());
  cl_context->setArg(2, TO_TENSOR_PTR(running_std_.get())->storage());
  cl_context->setArg(3, out->storage());
  if (affine_) {
    cl_context->setArg(4, TO_TENSOR_PTR(weights_.get())->storage());
    cl_context->setArg(5, TO_TENSOR_PTR(biases_.get())->storage());
  }
  cl_context->runKernel(jtorch::deviceid, TO_TENSOR_PTR(output.get())->dim(),
                        TO_TENSOR_PTR(output.get())->size(), false);
}

std::unique_ptr<TorchStage> SpatialBatchNormalization::loadFromFile(
    std::ifstream& file) {
  int32_t affine;
  file.read((char*)(&affine), sizeof(affine));
  int32_t nfeats;
  file.read((char*)(&nfeats), sizeof(nfeats));

  std::unique_ptr<SpatialBatchNormalization> ret(new SpatialBatchNormalization(
    affine == 1, nfeats));

  std::unique_ptr<float[]> data(new float[nfeats]);
  file.read((char*)(data.get()), sizeof(data[0]) * nfeats);
  ret->setRunningMean(data.get());

  file.read((char*)(data.get()), sizeof(data[0]) * nfeats);
  ret->setRunningStd(data.get());

  if (affine != 0) {
    file.read((char*)(data.get()), sizeof(data[0]) * nfeats);
    ret->setWeights(data.get());

    file.read((char*)(data.get()), sizeof(data[0]) * nfeats);
    ret->setBiases(data.get()); 
  }

  return std::unique_ptr<TorchStage>(std::move(ret));
}

void SpatialBatchNormalization::setWeights(const float* weights) {
  weights_->setData(weights);
}

void SpatialBatchNormalization::setBiases(const float* biases) {
  biases_->setData(biases);
}

void SpatialBatchNormalization::setRunningMean(const float* running_mean) {
  running_mean_->setData(running_mean);
}

void SpatialBatchNormalization::setRunningStd(const float* running_std) {
  running_std_->setData(running_std);
}

}  // namespace jtorch
