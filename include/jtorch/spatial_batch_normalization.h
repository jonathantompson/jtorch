//
//  spatial_batch_normalization.h
//
//  Created by Jonathan Tompson on 9/17/15.
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

class SpatialBatchNormalization : public TorchStage {
 public:
  // Constructor / Destructor
  SpatialBatchNormalization(const bool affine, const uint32_t nfeats);
  ~SpatialBatchNormalization() override;

  TorchStageType type() const override {
    return SPATIAL_BATCH_NORMALIZATION_STAGE;
  }
  std::string name() const override { return "SpatialBatchNormalization"; }
  void forwardProp(std::shared_ptr<TorchData> input) override;

  void setWeights(const float* weights);
  void setBiases(const float* biases);
  Tensor<float>* weights() { return weights_.get(); }
  Tensor<float>* biases() { return biases_.get(); }

  void setRunningMean(const float* running_mean);
  void setRunningStd(const float* running_std);
  Tensor<float>* running_mean() { return running_mean_.get(); }
  Tensor<float>* running_std() { return running_std_.get(); } 

  static std::unique_ptr<TorchStage> loadFromFile(std::ifstream& file);

 protected:
  bool affine_;
  uint32_t nfeats_;

  std::unique_ptr<Tensor<float>> weights_;
  std::unique_ptr<Tensor<float>> biases_;
  std::unique_ptr<Tensor<float>> running_mean_;
  std::unique_ptr<Tensor<float>> running_std_;

  void init(std::shared_ptr<TorchData> input);

  // Non-copyable, non-assignable.
  SpatialBatchNormalization(SpatialBatchNormalization&);
  SpatialBatchNormalization& operator=(const SpatialBatchNormalization&);
};

};  // namespace jtorch
