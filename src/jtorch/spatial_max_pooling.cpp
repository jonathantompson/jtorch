#include "jtorch/spatial_max_pooling.h"
#include "jtorch/tensor.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"
#include "jcl/data_str/vector_managed.h"

#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }

using namespace jcl::threading;
using namespace jcl::math;
using namespace jcl::data_str;

namespace jtorch {

  SpatialMaxPooling::SpatialMaxPooling(const uint32_t poolsize_v, 
    const uint32_t poolsize_u) : TorchStage() {
    poolsize_v_ = poolsize_v;
    poolsize_u_ = poolsize_u;
    output = NULL;
  }

  SpatialMaxPooling::~SpatialMaxPooling() {
    SAFE_DELETE(output);
  }

  void SpatialMaxPooling::init(TorchData& input)  {
    if (input.type() != TorchDataType::TENSOR_DATA) {
      throw std::runtime_error("SpatialMaxPooling::init() - "
        "FloatTensor expected!");
    }
    Tensor<float>& in = (Tensor<float>&)input;
    if (in.dim() != 2 && in.dim() != 3) {
      throw std::runtime_error("Input dimension must be 2D or 3D!");
    }

    if (output != NULL && TO_TENSOR_PTR(output)->dim() != in.dim()) {
      // Input dimension has changed!
      SAFE_DELETE(output);
    }

    if (output != NULL) {
      // Check that the dimensions above the lowest 2 match
      for (uint32_t i = 2; i < in.dim() && output != NULL; i++) {
        if (TO_TENSOR_PTR(output)->size()[i] != in.size()[i]) {
          SAFE_DELETE(output);
        }
      }
    }

    if (output != NULL) {
      // Check that the lowest 2 dimensions are the correct size
      if (TO_TENSOR_PTR(output)->size()[0] != in.size()[0] / poolsize_u_ ||
        TO_TENSOR_PTR(output)->size()[1] != in.size()[1] / poolsize_v_) {
        SAFE_DELETE(output);
      }
    }

    if (output == NULL) {
      if (in.size()[0] % poolsize_u_ != 0 || 
          in.size()[1] % poolsize_v_ != 0) {
        throw std::runtime_error("width or height is not a multiple of "
          "the poolsize!");
      }
      uint32_t* out_size = new uint32_t[in.dim()];
      out_size[0] = in.size()[0] / poolsize_u_;
      out_size[1] = in.size()[1] / poolsize_v_;
      for (uint32_t i = 2; i < in.dim(); i++) {
        out_size[i] = in.size()[i];
      }
      output = new Tensor<float>(in.dim(), out_size);
      SAFE_DELETE_ARR(out_size);
    }
  }

  void SpatialMaxPooling::forwardProp(TorchData& input) { 
    init(input);
    std::string kernel = jtorch::jtorch_path + "kernels/spatial_max_pooling.cl";
    bool two_dim = ((Tensor<float>&)input).dim() == 2;
    if (two_dim) {
      cl_context->useKernel(kernel.c_str(), "SpatialMaxPooling2D");
    } else {
      cl_context->useKernel(kernel.c_str(), "SpatialMaxPooling");
    }
    cl_context->setArg(0, ((Tensor<float>&)input).storage());
    cl_context->setArg(1, TO_TENSOR_PTR(output)->storage());
    cl_context->setArg(2, (int)((Tensor<float>&)input).size()[1]);
    cl_context->setArg(3, (int)((Tensor<float>&)input).size()[0]);
    cl_context->setArg(4, (int)poolsize_v_);
    cl_context->setArg(5, (int)poolsize_u_);
    cl_context->runKernel(jtorch::deviceid, TO_TENSOR_PTR(output)->dim(),
      TO_TENSOR_PTR(output)->size(), false);
  }

  TorchStage* SpatialMaxPooling::loadFromFile(std::ifstream& file) {
    int poolu, poolv;
    file.read((char*)(&poolu), sizeof(poolu));
    file.read((char*)(&poolv), sizeof(poolv));
    return new SpatialMaxPooling(poolu, poolv);
  }

}  // namespace jtorch