#include "jtorch/identity.h"

#include "jtorch/table.h"
#include "jtorch/tensor.h"

using namespace jcl::threading;
using namespace jcl::math;

namespace jtorch {

Identity::Identity() { output = nullptr; }

Identity::~Identity() {
  // Nothing to do for output (we don't own it)
}

std::unique_ptr<TorchStage> Identity::loadFromFile(std::ifstream& file) {
  // Nothing to load from file
  return std::unique_ptr<TorchStage>(new Identity());
}

void Identity::forwardProp(std::shared_ptr<TorchData> input) { output = input; }

}  // namespace jtorch
