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
using namespace jcl::math;
using namespace jcl::data_str;
using namespace jcl;

namespace jtorch {

  SpatialUpSamplingNearest::SpatialUpSamplingNearest(const int32_t scale) 
    : TorchStage() {
    scale_ = scale;
    output = NULL;
  }

  SpatialUpSamplingNearest::~SpatialUpSamplingNearest() {
    SAFE_DELETE(output);
  }

  void SpatialUpSamplingNearest::init(TorchData& input)  {
    if (input.type() != TorchDataType::TENSOR_DATA) {
      throw std::runtime_error("SpatialConvolution::init() - "
        "FloatTensor expected!");
    }

    Tensor<float>& in = (Tensor<float>&)input;
    Int3 out_dim(in.dim());
    out_dim[0] *= scale_;
    out_dim[1] *= scale_;
   
    if (output != NULL) {
      if (!Int3::equal(out_dim, ((Tensor<float>*)output)->dim())) {
        // Input dimension has changed!
        SAFE_DELETE(output);
      }
    }

    if (output == NULL) {
      output = new Tensor<float>(out_dim);
      //cl_context->getOptimalLocalWorkgroupSizes(deviceid, 
      //  ((Tensor<float>*)output)->dim(), local_worgroup_size);
    }
  }

  void SpatialUpSamplingNearest::forwardProp(TorchData& input) { 
    init(input);

    Tensor<float>& in = (Tensor<float>&)input;
    std::string kernel = jtorch::jtorch_path + "kernels/spatial_up_sampling_nearest.cl";
    cl_context->useKernel(kernel.c_str(), "SpatialUpSamplingNearest");
    cl_context->setArg(0, ((Tensor<float>&)input).data());
    cl_context->setArg(1, ((Tensor<float>*)output)->data());
    cl_context->setArg(2, scale_);
    cl_context->runKernel3D(jtorch::deviceid, ((Tensor<float>*)output)->dim(),
      false);
  }

  TorchStage* SpatialUpSamplingNearest::loadFromFile(std::ifstream& file) {
    int32_t scale;
    file.read((char*)(&scale), sizeof(scale));

    return new SpatialUpSamplingNearest(scale);
  }

}  // namespace jtorch