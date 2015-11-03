//
//  spatial_convolution_mm.h
//
//  Created by Jonathan Tompson on 2/4/15.
//

#pragma once

#include "jcl/math/int_types.h"
#include "jcl/math/math_types.h"
#include "jtorch/torch_stage.h"

namespace jtorch {

template <typename T>
class Tensor;

class SpatialConvolutionMM : public TorchStage {
 public:
  // Constructor / Destructor
  SpatialConvolutionMM(const uint32_t feats_in, const uint32_t feats_out,
                       const uint32_t filt_height, const uint32_t filt_width,
                       const uint32_t padw, const uint32_t padh);
  ~SpatialConvolutionMM() override;

  TorchStageType type() const override { return SPATIAL_CONVOLUTION_MM_STAGE; }
  std::string name() const override { return "SpatialConvolutionMM"; }
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
  uint32_t padw_;
  uint32_t padh_;

  std::unique_ptr<Tensor<float>> weights_;
  std::unique_ptr<Tensor<float>> biases_;

  std::unique_ptr<Tensor<float>>
      columns_;  // This is finput in torch.  TODO: Share this!
  std::unique_ptr<Tensor<float>> ones_;  // This is fgradinput in torch

  void init(std::shared_ptr<TorchData> input);

  // Non-copyable, non-assignable.
  SpatialConvolutionMM(const SpatialConvolutionMM&) = delete;
  SpatialConvolutionMM& operator=(const SpatialConvolutionMM&) = delete;
};

};  // namespace jtorch
