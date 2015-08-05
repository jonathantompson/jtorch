#include <sstream>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include "jtorch/torch_data.h"

#define SAFE_DELETE(x) if (x != nullptr) { delete x; x = nullptr; }
#define SAFE_DELETE_ARR(x) if (x != nullptr) { delete[] x; x = nullptr; }

namespace jtorch {

  TorchData::TorchData() {
  }

  TorchData::~TorchData() {
  }

}  // namespace jtorch