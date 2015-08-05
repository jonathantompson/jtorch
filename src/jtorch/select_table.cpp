#include "jtorch/select_table.h"
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

  SelectTable::SelectTable(int32_t index) {
    output = nullptr;
    index_ = index;
  }

  SelectTable::~SelectTable() {
    // Nothing to do for output (we don't own it)
  }


  TorchStage* SelectTable::loadFromFile(std::ifstream& file) {
    int32_t index;
    file.read((char*)(&index), sizeof(index));
    index = index - 1;  // We index from 0 in C++
    return new SelectTable(index);
  }

  void SelectTable::forwardProp(TorchData& input) {
    if (input.type() != TorchDataType::TABLE_DATA) {
      throw std::runtime_error("SelectTable::forwardProp() - "
        "Table expected!");
    }

    Table& in = (Table&)input;

    if ((int32_t)in.tableSize() <= index_) {
      throw std::runtime_error("SelectTable::forwardProp() - "
        "Input table is too small!");
    }

    output = in(index_);
  }

}  // namespace jtorch
