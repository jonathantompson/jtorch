#include "jtorch/spatial_up_sampling_nearest.h"
#include "jtorch/tensor.h"
#include "jtorch/jtorch.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"
#include "jcl/data_str/vector_managed.h"

#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }

using namespace jcl::threading;
using namespace jcl::data_str;
using namespace jcl;

namespace jtorch {

  SpatialUpSamplingNearest::SpatialUpSamplingNearest(const int32_t scale) 
    : TorchStage() {
    scale_ = scale;
    output = NULL;
    out_size_ = NULL;
  }

  SpatialUpSamplingNearest::~SpatialUpSamplingNearest() {
    SAFE_DELETE(output);
    SAFE_DELETE_ARR(out_size_);
  }

  void SpatialUpSamplingNearest::init(TorchData& input)  {
    if (input.type() != TorchDataType::TENSOR_DATA) {
      throw std::runtime_error("SpatialConvolution::init() - "
        "FloatTensor expected!");
    }

    Tensor<float>& in = (Tensor<float>&)input;
    Tensor<float>* out = (Tensor<float>*)output;

    if (in.dim() < 2) {
      throw std::runtime_error("SpatialConvolution::init() - "
        "Input must be 2D or larger!");
    }

    if (output != NULL && in.dim() != out->dim()) {
      SAFE_DELETE(output);
    }

    // Check that the inner 2 dimensions differ by a single scale
    if (output != NULL) {
      if (in.size()[0] * scale_ != out->size()[0] || 
          in.size()[1] * scale_ != out->size()[1]) {
        SAFE_DELETE(output);
      }
    }

    // Check that the remaining dimensions are the same size
    if (output != NULL) {
      for (uint32_t i = 2; i < in.dim() && output != NULL; i++) {
        if (in.size()[i] != out->size()[i]) {
          SAFE_DELETE(output);
        }
      }
    }

    if (output == NULL) {
      uint32_t* out_size = new uint32_t[in.dim()];
      memcpy(out_size, in.size(), sizeof(out_size[0]) * in.dim());
      out_size[0] *= scale_;
      out_size[1] *= scale_;

      output = new Tensor<float>(in.dim(), out_size);
      
      SAFE_DELETE_ARR(out_size);
    }
  }

  void SpatialUpSamplingNearest::forwardProp(TorchData& input) { 
    init(input);

    Tensor<float>& in = (Tensor<float>&)input;
    std::string kernel = jtorch::jtorch_path + "kernels/spatial_up_sampling_nearest.cl";
    if (in.dim() == 2) {
      cl_context->useKernel(kernel.c_str(), "SpatialUpSamplingNearest2D");
    } else {
      cl_context->useKernel(kernel.c_str(), "SpatialUpSamplingNearest");
    }
    cl_context->setArg(0, ((Tensor<float>&)input).storage());
    cl_context->setArg(1, TO_TENSOR_PTR(output)->storage());
    cl_context->setArg(2, (int)scale_);
    cl_context->runKernel(jtorch::deviceid, TO_TENSOR_PTR(output)->dim(),
      TO_TENSOR_PTR(output)->size(), false);
  }

  TorchStage* SpatialUpSamplingNearest::loadFromFile(std::ifstream& file) {
    int32_t scale;
    file.read((char*)(&scale), sizeof(scale));

    return new SpatialUpSamplingNearest(scale);
  }

}  // namespace jtorch