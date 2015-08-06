//
//  table.h
//
//  Created by Jonathan Tompson on 4/1/13.
//
//  Simplified C++ replica of a lua table.  It was implemented so that parallel
//  modules work.
//

#pragma once

#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include "jcl/math/int_types.h"
#include "jtorch/torch_data.h"

#define TO_TABLE_PTR(x) \
  (x->type() == jtorch::TorchDataType::TABLE_DATA ? (jtorch::Table*)x : nullptr)

namespace jtorch {

class Table : public TorchData {
 public:
  // Constructor / Destructor
  Table();  // Create an empty table
  ~Table() override;

  std::shared_ptr<TorchData> operator()(const uint32_t i);
  void add(std::shared_ptr<TorchData> new_data);

  void clear();

  TorchDataType type() const override { return TABLE_DATA; }
  void print() override;  // print to std::cout

  uint32_t tableSize() const;

 protected:
  std::vector<std::shared_ptr<TorchData>> data_;  // Internal data

  // Non-copyable, non-assignable.
  Table(Table&);
  Table& operator=(const Table&);
};

};  // namespace jtorch
