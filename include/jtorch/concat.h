//
//  concat.h
//
//  Created by Jonathan Tompson on 11/3/15.
//

#pragma once

#include "jcl/math/int_types.h"
#include "jcl/math/math_types.h"
#include "jtorch/torch_stage.h"

namespace jtorch {

class Concat : public TorchStage {
 public:
  // Constructor / Destructor
  Concat(int dimension);
  ~Concat() override;

  TorchStageType type() const override { return CONCAT_STAGE; }
  std::string name() const override { return "Concat"; }
  void forwardProp(std::shared_ptr<TorchData> input) override;

  void add(std::unique_ptr<TorchStage> stage);
  TorchStage* get(const uint32_t i);
  uint32_t size() const;

  static std::unique_ptr<TorchStage> loadFromFile(std::ifstream& file);

 protected:
  int dimension_;

  std::vector<std::unique_ptr<TorchStage>> network_;

  // Non-copyable, non-assignable.
  Concat(const Concat&) = delete;
  Concat& operator=(const Concat&) = delete;
};

};  // namespace jtorch
