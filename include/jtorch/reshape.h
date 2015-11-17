//
//  reshape.h
//
//  Created by Jonathan Tompson on 4/1/13.
//
//  Works a little differently to the torch version...  It takes a 3D tensor
//  and just makes it into a 1D array, so it's not as general purpose.
//
//  But really this 1D array is just a straight copy of the input data (since
//  we define tensors as float* anyway).
//

#pragma once

#include "jcl/math/math_types.h"
#include "jtorch/torch_stage.h"

namespace jtorch {

class Reshape : public TorchStage {
 public:
  // Constructor / Destructor
  // For 1D tensor: set sz1 = -1, for 2D tensor: set sz2 = -1
  Reshape(const uint32_t dim, const uint32_t* size);
  ~Reshape() override;

  TorchStageType type() const override { return RESHAPE_STAGE; }
  std::string name() const override { return "Reshape"; }
  void forwardProp(std::shared_ptr<TorchData> input) override;

  static std::unique_ptr<TorchStage> loadFromFile(std::ifstream& file);

 protected:
  uint32_t odim_;
  std::unique_ptr<uint32_t[]> osize_;
  void init(std::shared_ptr<TorchData> input);

  uint32_t outNElem() const;

  // Non-copyable, non-assignable.
  Reshape(const Reshape&) = delete;
  Reshape& operator=(const Reshape&) = delete;
};

};  // namespace jtorch
