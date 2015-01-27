#include "jtorch/select_table.h"
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

  SelectTable::SelectTable() {
    output = NULL;
  }

  SelectTable::~SelectTable() {
    // Nothing to do for output (we don't own it)
  }


  TorchStage* SelectTable::loadFromFile(std::ifstream& file) {
    SelectTable* module = new SelectTable();
    file.read((char*)(&module->index_), sizeof(module->index_));
    return module;
  }

  void SelectTable::forwardProp(TorchData& input) {
    if (input.type() != TorchDataType::TABLE_DATA) {
      throw std::runtime_error("SelectTable::forwardProp() - "
        "Table expected!");
    }

    Table& in = (Table&)input;

    if (in.tableSize() <= index_) {
      throw std::runtime_error("SelectTable::forwardProp() - "
        "Input table is too small!");
    }

    output = in(index_);
  }

}  // namespace jtorch
