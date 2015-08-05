#include "jtorch/identity.h"
#include "jtorch/tensor.h"
#include "jtorch/table.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"
#include "jcl/data_str/vector_managed.h"

#define SAFE_DELETE(x) if (x != nullptr) { delete x; x = nullptr; }
#define SAFE_DELETE_ARR(x) if (x != nullptr) { delete[] x; x = nullptr; }

using namespace jcl::threading;
using namespace jcl::math;
using namespace jcl::data_str;

namespace jtorch {

  Identity::Identity() {
    output = nullptr;
  }

  Identity::~Identity() {
    // Nothing to do for output (we don't own it)
  }

  TorchStage* Identity::loadFromFile(std::ifstream& file) {
    // Nothing to load for identity
    return new Identity();
  }

  void Identity::forwardProp(TorchData& input) {
    output = &input;
  }

}  // namespace jtorch