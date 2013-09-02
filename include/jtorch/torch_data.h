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
#include "jtil/math/math_types.h"

namespace jtil { namespace threading { class ThreadPool; } }

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
    virtual uint32_t dataSize() const = 0;  // Pure virtual
    virtual void print() = 0;

  protected:
    // Non-copyable, non-assignable.
    TorchData(TorchData&);
    TorchData& operator=(const TorchData&);
  };

};  // namespace jtorch
