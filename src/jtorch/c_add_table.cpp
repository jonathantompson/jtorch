#include "jtorch/c_add_table.h"
#include "jtorch/tensor.h"
#include "jtorch/table.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"
#include "jcl/data_str/vector_managed.h"
#include "jcl/jcl.h"

using namespace jcl::threading;
using namespace jcl::math;
using namespace jcl::data_str;

namespace jtorch {

  CAddTable::CAddTable() {
    output = nullptr;
  }

  CAddTable::~CAddTable() {
  }


  std::unique_ptr<TorchStage> CAddTable::loadFromFile(std::ifstream& file) {
    // Nothing to load from file
    return std::unique_ptr<TorchStage>(new CAddTable());
  }

  void CAddTable::forwardProp(std::shared_ptr<TorchData> input) {
    assert(input->type() == TorchDataType::TABLE_DATA);

    Table* in = reinterpret_cast<Table*>(input.get());
    assert(in->tableSize() != 0);

    // Make sure all the elements are of type TENSOR
    for (uint32_t i = 0; i < in->tableSize(); i++) {
      // Table of Tensors expected.
      assert((*in)(i)->type() == TorchDataType::TENSOR_DATA);
    }

    for (uint32_t i = 1; i < in->tableSize(); i++) {
      // Table of equal size tensors expected.
      assert(TO_TENSOR_PTR((*in)(0).get())->isSameSizeAs(*TO_TENSOR_PTR((*in)(i).get())));
    }

    if (output == nullptr || 
      !TO_TENSOR_PTR((*in)(0).get())->isSameSizeAs(*TO_TENSOR_PTR(output.get()))) {
      // Reinitialize the output Tensor
      output.reset(new Tensor<float>(TO_TENSOR_PTR((*in)(0).get())->dim(), 
        TO_TENSOR_PTR((*in)(0).get())->size()));
    }

    // TODO: We can probabily parallelize these calls accross multiple tensors
    Tensor<float>::copy(*TO_TENSOR_PTR(output.get()), *TO_TENSOR_PTR((*in)(0).get()));
    for (uint32_t i = 1; i < in->tableSize(); i++) {
      Tensor<float>::accumulate(*TO_TENSOR_PTR(output.get()), *TO_TENSOR_PTR((*in)(i).get()));
    }

  }

}  // namespace jtorch
