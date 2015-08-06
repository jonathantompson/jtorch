#include <sstream>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include "jtorch/table.h"

namespace jtorch {

Table::Table() {}

Table::~Table() {}

std::shared_ptr<TorchData> Table::operator()(const uint32_t i) {
  return data_[i];
}

void Table::print() {
  for (uint32_t i = 0; i < data_.size(); i++) {
    std::cout << "Table[" << i << "] = " << std::endl;
    data_[i]->print();
  }
};

void Table::add(std::shared_ptr<TorchData> new_data) {
  data_.push_back(new_data);
}

void Table::clear() {
  for (uint32_t i = 0; i < data_.size(); i++) {
    data_[i] = nullptr;
  }
  data_.resize(0);
}

uint32_t Table::tableSize() const { return (uint32_t)data_.size(); }

}  // namespace jtorch
