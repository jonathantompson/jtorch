#include "jtorch/transpose.h"
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

  Transpose::Transpose() {
    output = nullptr;
  }

  Transpose::~Transpose() {
  }

  std::unique_ptr<TorchStage> Transpose::loadFromFile(std::ifstream& file) {
    int32_t num_permutations;
    file.read((char*)(&num_permutations), sizeof(num_permutations));
    std::unique_ptr<int32_t[]> perms(new int32_t[num_permutations * 2]);
    file.read((char*)(perms.get()), sizeof(perms[0]) * num_permutations * 2);
    // But we don't really use it...
    return std::unique_ptr<TorchStage>(new Transpose());
  }

  void Transpose::forwardProp(std::shared_ptr<TorchData> input) {
    output = input;
  }

}  // namespace jtorch