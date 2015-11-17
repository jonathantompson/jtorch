#include "jtorch/select_table.h"

#include "jtorch/table.h"
#include "jtorch/tensor.h"

using namespace jcl::threading;
using namespace jcl::math;

namespace jtorch {

SelectTable::SelectTable(int32_t index) {
  output = nullptr;
  index_ = index;
}

SelectTable::~SelectTable() {
  // Nothing to do for output (we don't own it)
}

std::unique_ptr<TorchStage> SelectTable::loadFromFile(std::ifstream& file) {
  int32_t index;
  file.read((char*)(&index), sizeof(index));
  index = index - 1;  // We index from 0 in C++
  return std::unique_ptr<TorchStage>(new SelectTable(index));
}

void SelectTable::forwardProp(std::shared_ptr<TorchData> input) {
  RASSERT(input->type() == TorchDataType::TABLE_DATA);

  Table* in = (Table*)input.get();

  // Check that the input table isn't too small.
  RASSERT(in->tableSize() > index_);

  output = (*in)(index_);
}

}  // namespace jtorch
