#include <sstream>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include "jtorch/torch_data.h"
#include "jtil/exceptions/wruntime_error.h"

#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }

namespace jtorch {

  TorchData::TorchData() {
  }

  TorchData::~TorchData() {
  }

}  // namespace jtorch