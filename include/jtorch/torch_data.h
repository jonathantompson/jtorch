//
//  torch_data.h
//
//  Created by Jonathan Tompson on 4/1/13.
//
//  This is the base class that other data classes derive from.
//

#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include "jcl/math/int_types.h"

namespace jtorch {

typedef enum {
  UNDEFINED_DATA = 0,
  TABLE_DATA = 1,
  TENSOR_DATA = 2,
} TorchDataType;

class TorchData {
 public:
  // Constructor / Destructor
  TorchData();
  virtual ~TorchData();

  virtual TorchDataType type() const { return UNDEFINED_DATA; }
  virtual void print() = 0;

 protected:
  // Non-copyable, non-assignable.
  TorchData(TorchData&);
  TorchData& operator=(const TorchData&);
};

};  // namespace jtorch
