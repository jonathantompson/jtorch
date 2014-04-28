#include "jtorch/transpose.h"
#include "jtorch/tensor.h"
#include "jtorch/table.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"
#include "jcl/data_str/vector_managed.h"

#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }

using namespace jcl::threading;
using namespace jcl::math;
using namespace jcl::data_str;

namespace jtorch {

  Transpose::Transpose() {
    output = NULL;
  }

  Transpose::~Transpose() {
  }

  TorchStage* Transpose::loadFromFile(std::ifstream& file) {
    int32_t num_permutations;
    file.read((char*)(&num_permutations), sizeof(num_permutations));
    int32_t* perms = new int32_t[num_permutations * 2];
    file.read((char*)(perms), sizeof(perms[0]) * num_permutations * 2);
    // But we don't really use it...
    TorchStage* ret_val = new Transpose();
    delete[] perms;
    return ret_val;
  }

  void Transpose::forwardProp(TorchData& input) {
    output = &input;
  }

}  // namespace jtorch