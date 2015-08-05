#include "jtorch/c_add_table.h"
#include "jtorch/tensor.h"
#include "jtorch/table.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"
#include "jcl/data_str/vector_managed.h"
#include "jcl/jcl.h"

#define SAFE_DELETE(x) if (x != nullptr) { delete x; x = nullptr; }
#define SAFE_DELETE_ARR(x) if (x != nullptr) { delete[] x; x = nullptr; }

using namespace jcl::threading;
using namespace jcl::math;
using namespace jcl::data_str;

namespace jtorch {

  CAddTable::CAddTable() {
    output = nullptr;
  }

  CAddTable::~CAddTable() {
    delete output;
  }


  TorchStage* CAddTable::loadFromFile(std::ifstream& file) {
    // Nothing to load from file
    return new CAddTable();
  }

  void CAddTable::forwardProp(TorchData& input) {
    if (input.type() != TorchDataType::TABLE_DATA) {
      throw std::runtime_error("SelectTable::forwardProp() - "
        "Table expected!");
    }

    Table& in = (Table&)input;

    if (in.tableSize() == 0) {
      throw std::runtime_error("SelectTable::forwardProp() - "
        "Input table is empty!");
    }

    // Make sure all the elements are of type TENSOR
    for (uint32_t i = 0; i < in.tableSize(); i++) {
      if (in(i)->type() != TorchDataType::TENSOR_DATA) {
        throw std::runtime_error("SelectTable::forwardProp() - "
          "Table of Tensors expected!");
      }
    }

    for (uint32_t i = 1; i < in.tableSize(); i++) {
      if (!TO_TENSOR_PTR(in(0))->isSameSizeAs(*TO_TENSOR_PTR(in(i)))) {
        throw std::runtime_error("SelectTable::forwardProp() - "
          "Table of equal size Tensors expected!");
      }
    }

    if (output == nullptr || 
      !TO_TENSOR_PTR(in(0))->isSameSizeAs(*TO_TENSOR_PTR(output))) {
      // Reinitialize the output Tensor
      SAFE_DELETE(output);
      output = new Tensor<float>(TO_TENSOR_PTR(in(0))->dim(), 
        TO_TENSOR_PTR(in(0))->size());
    }

    // TODO: We can probabily parallelize these calls accross multiple tensors
    Tensor<float>::copy(*TO_TENSOR_PTR(output), *TO_TENSOR_PTR(in(0)));
    for (uint32_t i = 1; i < in.tableSize(); i++) {
      Tensor<float>::accumulate(*TO_TENSOR_PTR(output), *TO_TENSOR_PTR(in(i)));
    }

  }

}  // namespace jtorch
