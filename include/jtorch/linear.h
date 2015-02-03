//
//  linear.h
//
//  Created by Jonathan Tompson on 4/1/13.
//

#pragma once

#include <mutex>
#include <condition_variable>
#include "jcl/math/int_types.h"
#include "jtorch/torch_stage.h"

#define SIMPLE_LINEAR  // Might actually be faster when using the CPU!

namespace jtorch {

  template <typename T> class Tensor;
  
  class Linear : public TorchStage {
  public:
    // Constructor / Destructor
    Linear(const uint32_t n_inputs, const uint32_t n_outputs);
    virtual ~Linear();

    virtual TorchStageType type() const { return LINEAR_STAGE; }
    virtual void forwardProp(TorchData& input);

    void setWeights(const float* weights);
    void setBiases(const float* biases);
    Tensor<float>* weights() { return weights_; }
    Tensor<float>* biases() { return biases_; }

    static TorchStage* loadFromFile(std::ifstream& file);

  protected:
    uint32_t n_inputs_;
    uint32_t n_outputs_;

    Tensor<float>* weights_;  // n_outputs (rows) * n_inputs (columns), stored row major
    Tensor<float>* biases_;  // n_outputs

    void init(TorchData& input);

    // Non-copyable, non-assignable.
    Linear(Linear&);
    Linear& operator=(const Linear&);
  };
  
};  // namespace jtorch
