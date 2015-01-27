#include "jtorch/c_add_table.h"
#include "jtorch/tensor.h"
#include "jtorch/table.h"
#include "jcl/threading/thread.h"
#include "jcl/threading/callback.h"
#include "jcl/threading/thread_pool.h"
#include "jcl/data_str/vector_managed.h"
#include "jcl/jcl.h"

#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }

using namespace jcl::threading;
using namespace jcl::math;
using namespace jcl::data_str;

namespace jtorch {

  CAddTable::CAddTable() {
    output = NULL;
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

    const Int3& dim = ((Tensor<float>*)in(0))->dim();
    for (uint32_t i = 1; i < in.tableSize(); i++) {
      if (!Int3::equal(dim, ((Tensor<float>*)in(i))->dim())) {
        throw std::runtime_error("SelectTable::forwardProp() - "
          "Table of equal size Tensors expected!");
      }
    }

    if (output == NULL || !Int3::equal(dim, ((Tensor<float>*)output)->dim())) {
      // Reinitialize the output Tensor
      SAFE_DELETE(output);
      output = new Tensor<float>(dim);
    }

    // TODO: We can probabily parallelize these calls accross multiple tensors
    Tensor<float>* out = (Tensor<float>*)output;
    Tensor<float>::copy(*out, *(Tensor<float>*)in(0));
    for (uint32_t i = 1; i < in.tableSize(); i++) {
      Tensor<float>::accumulate(*out, *(Tensor<float>*)in(i));
    }

  }

}  // namespace jtorch
