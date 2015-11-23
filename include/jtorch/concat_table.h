//
//  concat_table.h
//
//  Created by Jonathan Tompson on 11/23/15.
//

#pragma once

#include "jcl/math/int_types.h"
#include "jcl/math/math_types.h"
#include "jtorch/torch_stage.h"

namespace jtorch {

class ConcatTable : public TorchStage {
 public:
  // Constructor / Destructor
  ConcatTable();
  ~ConcatTable() override;

  TorchStageType type() const override { return CONCAT_TABLE_STAGE; }
  std::string name() const override { return "ConcatTable"; }
  void forwardProp(std::shared_ptr<TorchData> input) override;

  void add(std::unique_ptr<TorchStage> stage);
  TorchStage* get(const uint32_t i);
  uint32_t size() const;

  static std::unique_ptr<TorchStage> loadFromFile(std::ifstream& file);

 protected:
  std::vector<std::unique_ptr<TorchStage>> network_;

  // Non-copyable, non-assignable.
  ConcatTable(const ConcatTable&) = delete;
  ConcatTable& operator=(const ConcatTable&) = delete;
};

};  // namespace jtorch
