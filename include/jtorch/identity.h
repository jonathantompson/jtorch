//
//  identity.h
//
//  Created by Jonathan Tompson on 1/26/15.
//
//  Identity just passes the data through
//

#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include "jtorch/torch_stage.h"

namespace jtorch {

class Identity : public TorchStage {
 public:
  // Constructor / Destructor
  Identity();
  ~Identity() override;

  TorchStageType type() const override { return IDENTITY_STAGE; }
  std::string name() const override { return "Identity"; }
  void forwardProp(std::shared_ptr<TorchData> input) override;

  static std::unique_ptr<TorchStage> loadFromFile(std::ifstream& file);

 protected:
  // Non-copyable, non-assignable.
  Identity(Identity&);
  Identity& operator=(const Identity&);
};

};  // namespace jtorch
