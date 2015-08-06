#include "jtorch/spatial_max_pooling.h"
#include "jtorch/tensor.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"

using namespace jcl::threading;
using namespace jcl::math;

namespace jtorch {

  SpatialMaxPooling::SpatialMaxPooling(const uint32_t poolsize_v, 
    const uint32_t poolsize_u) : TorchStage() {
    poolsize_v_ = poolsize_v;
    poolsize_u_ = poolsize_u;
    output = nullptr;
  }

  SpatialMaxPooling::~SpatialMaxPooling() {
  }

  void SpatialMaxPooling::init(std::shared_ptr<TorchData> input)  {
    assert(input->type() == TorchDataType::TENSOR_DATA);
    Tensor<float>* in = TO_TENSOR_PTR(input.get());
    assert(in->dim() == 2 || in->dim() == 3);

    if (output != nullptr && TO_TENSOR_PTR(output.get())->dim() != in->dim()) {
      // Input dimension has changed!
      output = nullptr;
    }

    if (output != nullptr) {
      // Check that the dimensions above the lowest 2 match
      for (uint32_t i = 2; i < in->dim() && output != nullptr; i++) {
        if (TO_TENSOR_PTR(output.get())->size()[i] != in->size()[i]) {
          output = nullptr;
        }
      }
    }

    if (output != nullptr) {
      // Check that the lowest 2 dimensions are the correct size
      if (TO_TENSOR_PTR(output.get())->size()[0] != in->size()[0] / poolsize_u_ ||
        TO_TENSOR_PTR(output.get())->size()[1] != in->size()[1] / poolsize_v_) {
        output = nullptr;
      }
    }

    if (output == nullptr) {
      // Check that the width and height is a multiple of the poolsize
      assert(in->size()[0] % poolsize_u_ == 0 && 
             in->size()[1] % poolsize_v_ == 0);
      std::unique_ptr<uint32_t[]> out_size(new uint32_t[in->dim()]);
      out_size[0] = in->size()[0] / poolsize_u_;
      out_size[1] = in->size()[1] / poolsize_v_;
      for (uint32_t i = 2; i < in->dim(); i++) {
        out_size[i] = in->size()[i];
      }
      output.reset(new Tensor<float>(in->dim(), out_size.get()));
    }
  }

  void SpatialMaxPooling::forwardProp(std::shared_ptr<TorchData> input) { 
    init(input);
    std::string kernel = jtorch::jtorch_path + "kernels/spatial_max_pooling.cl";
    bool two_dim = TO_TENSOR_PTR(input.get())->dim() == 2;
    if (two_dim) {
      cl_context->useKernel(kernel.c_str(), "SpatialMaxPooling2D");
    } else {
      cl_context->useKernel(kernel.c_str(), "SpatialMaxPooling");
    }
    cl_context->setArg(0, TO_TENSOR_PTR(input.get())->storage());
    cl_context->setArg(1, TO_TENSOR_PTR(output.get())->storage());
    cl_context->setArg(2, (int)TO_TENSOR_PTR(input.get())->size()[1]);
    cl_context->setArg(3, (int)TO_TENSOR_PTR(input.get())->size()[0]);
    cl_context->setArg(4, (int)poolsize_v_);
    cl_context->setArg(5, (int)poolsize_u_);
    cl_context->runKernel(jtorch::deviceid, TO_TENSOR_PTR(output.get())->dim(),
      TO_TENSOR_PTR(output.get())->size(), false);
  }

  std::unique_ptr<TorchStage> SpatialMaxPooling::loadFromFile(std::ifstream& file) {
    int poolu, poolv;
    file.read((char*)(&poolu), sizeof(poolu));
    file.read((char*)(&poolv), sizeof(poolv));
    return std::unique_ptr<TorchStage>(new SpatialMaxPooling(poolu, poolv));
  }

}  // namespace jtorch
