//
//  transpose.h
//
//  Created by Jonathan Tompson on 4/9/13.
//
//  NOTE: Transpose is NOT implemented, it just passes the data through
//

#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include "jtorch/torch_stage.h"

namespace jtorch {

class Transpose : public TorchStage {
 public:
  // Constructor / Destructor
  Transpose();
  ~Transpose() override;

  TorchStageType type() const override { return TRANSPOSE_STAGE; }
  std::string name() const override { return "Transpose"; }
  void forwardProp(std::shared_ptr<TorchData> input) override;

  static std::unique_ptr<TorchStage> loadFromFile(std::ifstream& file);

 protected:
  // Non-copyable, non-assignable.
  Transpose(Transpose&);
  Transpose& operator=(const Transpose&);
};

};  // namespace jtorch
