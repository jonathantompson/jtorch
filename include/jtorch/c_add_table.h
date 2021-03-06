//
//  c_add_table.h
//
//  Created by Jonathan Tompson on 1/27/15.
//

#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdint.h>

#include "jtorch/torch_stage.h"

namespace jtorch {

class CAddTable : public TorchStage {
 public:
  // Constructor / Destructor
  CAddTable();
  ~CAddTable() override;

  TorchStageType type() const override { return C_ADD_TABLE_STAGE; }
  std::string name() const override { return "CAddTable"; }
  void forwardProp(std::shared_ptr<TorchData> input) override;

  static std::unique_ptr<TorchStage> loadFromFile(std::ifstream& file);

 protected:
  // Non-copyable, non-assignable.
  CAddTable(const CAddTable&) = delete;
  CAddTable& operator=(const CAddTable&) = delete;
};

};  // namespace jtorch
