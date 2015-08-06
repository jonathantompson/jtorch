//
//  spatial_convolution.h
//
//  Created by Jonathan Tompson on 5/15/13.
//

#pragma once

#include <mutex>
#include <condition_variable>
#include "jcl/math/int_types.h"
#include "jcl/math/math_types.h"
#include "jtorch/torch_stage.h"
#include "jcl/jcl.h"  // For jcl::JCLBuffer

namespace jtorch {

template <typename T>
class Tensor;

class SpatialConvolution : public TorchStage {
 public:
  // Constructor / Destructor
  SpatialConvolution(const uint32_t feats_in, const uint32_t feats_out,
                     const uint32_t filt_height, const uint32_t filt_width,
                     const uint32_t padding = 0);
  ~SpatialConvolution() override;

  TorchStageType type() const override { return SPATIAL_CONVOLUTION_STAGE; }
  std::string name() const override { return "SpatialConvolution"; }
  void forwardProp(std::shared_ptr<TorchData> input) override;

  void setWeights(const float* weights);
  void setBiases(const float* biases);
  Tensor<float>* weights() { return weights_.get(); }
  Tensor<float>* biases() { return biases_.get(); }

  static std::unique_ptr<TorchStage> loadFromFile(std::ifstream& file);

 protected:
  uint32_t filt_width_;
  uint32_t filt_height_;
  uint32_t feats_in_;
  uint32_t feats_out_;
  uint32_t padding_;

  std::unique_ptr<Tensor<float>> weights_;
  std::unique_ptr<Tensor<float>> biases_;

  void init(std::shared_ptr<TorchData> input);

  // Non-copyable, non-assignable.
  SpatialConvolution(SpatialConvolution&);
  SpatialConvolution& operator=(const SpatialConvolution&);
};

};  // namespace jtorch
