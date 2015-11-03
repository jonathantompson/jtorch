//
//  select_table.h
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

class SelectTable : public TorchStage {
 public:
  // Constructor / Destructor
  SelectTable(int32_t index);
  ~SelectTable() override;

  TorchStageType type() const override { return SELECT_TABLE_STAGE; }
  std::string name() const override { return "SelectTable"; }
  void forwardProp(std::shared_ptr<TorchData> input) override;

  static std::unique_ptr<TorchStage> loadFromFile(std::ifstream& file);

 protected:
  uint32_t index_;

  // Non-copyable, non-assignable.
  SelectTable(const SelectTable&) = delete;
  SelectTable& operator=(const SelectTable&) = delete;
};

};  // namespace jtorch
