#include "jtorch/identity.h"
#include "jtorch/tensor.h"
#include "jtorch/table.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"
#include "jcl/data_str/vector_managed.h"

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

  std::unique_ptr<TorchStage> Identity::loadFromFile(std::ifstream& file) {
    // Nothing to load from file
    return std::unique_ptr<TorchStage>(new Identity());
  }

  void Identity::forwardProp(std::shared_ptr<TorchData> input) {
    output = input;
  }

}  // namespace jtorch