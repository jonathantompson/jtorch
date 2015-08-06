//
//  sequential.h
//
//  Created by Jonathan Tompson on 4/2/13.
//

#pragma once

#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include "jcl/math/int_types.h"
#include "jtorch/torch_stage.h"

namespace jcl {
namespace data_str {
template <typename T>
class VectorManaged;
}
}

namespace jtorch {

class Sequential : public TorchStage {
 public:
  // Constructor / Destructor
  Sequential();
  ~Sequential() override;

  TorchStageType type() const override { return SEQUENTIAL_STAGE; }
  std::string name() const override { return "Sequential"; }
  void forwardProp(std::shared_ptr<TorchData> input) override;

  void add(std::unique_ptr<TorchStage> stage);
  TorchStage* get(const uint32_t i);
  uint32_t size() const;

  static std::unique_ptr<TorchStage> loadFromFile(std::ifstream& file);

 protected:
  std::vector<std::unique_ptr<TorchStage>> network_;

  // Non-copyable, non-assignable.
  Sequential(Sequential&);
  Sequential& operator=(const Sequential&);
};

};  // namespace jtorch
