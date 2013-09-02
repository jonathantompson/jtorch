#include "jtorch/spatial_max_pooling.h"
#include "jtorch/tensor.h"
#include "jtil/exceptions/wruntime_error.h"
#include "jtil/threading/thread.h"
#include "jtil/threading/callback.h"
#include "jtil/threading/thread_pool.h"
#include "jtil/data_str/vector_managed.h"

#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }

using namespace jtil::threading;
using namespace jtil::math;
using namespace jtil::data_str;

namespace jtorch {

  SpatialMaxPooling::SpatialMaxPooling(const int32_t poolsize_v, 
    const int32_t poolsize_u) : TorchStage() {
    poolsize_v_ = poolsize_v;
    poolsize_u_ = poolsize_u;
    output = NULL;
  }

  SpatialMaxPooling::~SpatialMaxPooling() {
    SAFE_DELETE(output);
  }

  void SpatialMaxPooling::init(TorchData& input)  {
    if (input.type() != TorchDataType::TENSOR_DATA) {
      throw std::wruntime_error("SpatialMaxPooling::init() - "
        "FloatTensor expected!");
    }
    Tensor<float>& in = (Tensor<float>&)input;
    if (output != NULL) {
      if (!Int3::equal(in.dim(), ((Tensor<float>*)output)->dim())) {
        // Input dimension has changed!
        SAFE_DELETE(output);
      }
    }
    if (output == NULL) {
      if (in.dim()[0] % poolsize_u_ != 0 || 
        in.dim()[1] % poolsize_v_ != 0) {
        throw std::wruntime_error("width or height is not a multiple of "
          "the poolsize!");
      }
      Int3 out_dim(in.dim());
      out_dim[0] /= poolsize_u_;
      out_dim[1] /= poolsize_v_;
      output = new Tensor<float>(out_dim);

      //cl_context->getOptimalLocalWorkgroupSizes(deviceid, 
      //  ((Tensor<float>*)output)->dim(), local_worgroup_size);
    }
  }

  void SpatialMaxPooling::forwardProp(TorchData& input) { 
    init(input);
    std::string kernel = jtorch::jtorch_path + "kernels/spatial_max_pooling.cl";
    cl_context->useKernel(kernel.c_str(), "SpatialMaxPooling");
    cl_context->setArg(0, ((Tensor<float>&)input).data());
    cl_context->setArg(1, ((Tensor<float>*)output)->data());
    cl_context->setArg(2, ((Tensor<float>&)input).dim()[1]);
    cl_context->setArg(3, ((Tensor<float>&)input).dim()[0]);
    cl_context->setArg(4, poolsize_v_);
    cl_context->setArg(5, poolsize_u_);
    cl_context->runKernel3D(jtorch::deviceid, ((Tensor<float>*)output)->dim(),
      false);
  }

  TorchStage* SpatialMaxPooling::loadFromFile(std::ifstream& file) {
    int poolu, poolv;
    file.read((char*)(&poolu), sizeof(poolu));
    file.read((char*)(&poolv), sizeof(poolv));
    return new SpatialMaxPooling(poolu, poolv);
  }

}  // namespace jtorch