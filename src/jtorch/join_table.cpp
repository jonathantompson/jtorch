#include "jtorch/join_table.h"
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

  JoinTable::JoinTable(const uint32_t dimension) {
    dimension_ = dimension;
    output = nullptr;
  }

  JoinTable::~JoinTable() {
  }


  std::unique_ptr<TorchStage> JoinTable::loadFromFile(std::ifstream& file) {
    int32_t dimension;
    file.read((char*)(&dimension), sizeof(dimension));
    dimension = dimension - 1;  // We index from 0 in C++
    // But we don't really use it...
    return std::unique_ptr<TorchStage>(new JoinTable(dimension));
  }

  void JoinTable::init(std::shared_ptr<TorchData> input) {
    assert(input->type() == TorchDataType::TABLE_DATA);  // Table expected

    Table* in = TO_TABLE_PTR(input.get());

    assert(in->tableSize() > 0);

    // Check that it is a table of FloatTensors
    for (uint32_t i = 0; i < in->tableSize(); i++) {
      // Table of float tensors expected
      assert((*in)(i)->type() == TENSOR_DATA);
    }

    uint32_t dim = TO_TENSOR_PTR((*in)(0).get())->dim();
    assert(dim > dimension_);  // Otherwise input is smaller than join dimension
    uint32_t jdim = dim - dimension_ - 1;  // dimension_=0 is the top dim

    // Make sure the dimensions OTHER than the join dimension are all the same
    for (uint32_t d = 0; d < dim; d++) {
      if (d != jdim) {
        for (uint32_t j = 1; j < in->tableSize(); j++) {
          // sizes must match
          assert(TO_TENSOR_PTR((*in)(j).get())->size()[d] == 
                 TO_TENSOR_PTR((*in)(0).get())->size()[d]);
        }
        if (output != nullptr && TO_TENSOR_PTR(output.get())->size()[d] != 
          TO_TENSOR_PTR((*in)(0).get())->size()[d]) {
          output = nullptr;
        }
      }
    }

    uint32_t nelems_jdim = 0;
    for (uint32_t j = 1; j < in->tableSize(); j++) {
      nelems_jdim += TO_TENSOR_PTR((*in)(j).get())->size()[jdim];
    }

    if (output != nullptr &&
        TO_TENSOR_PTR(output.get())->size()[jdim] != nelems_jdim) {
      output = nullptr;
    }

    if (output == nullptr) {
      std::unique_ptr<uint32_t[]> size(new uint32_t[dim]);
      memcpy(size.get(), TO_TENSOR_PTR((*in)(0).get())->size(), sizeof(size[0]) * dim);
      size[dimension_] = nelems_jdim;
      output = std::shared_ptr<TorchData>(new Tensor<float>(dim, size.get()));
    }
  }

  void JoinTable::forwardProp(std::shared_ptr<TorchData> input) {
    init(input);

    Table* in = (Table*)input.get();

    // AT THE MOMENT ONLY JOINS ALONG THE TOP DIMENSION ARE SUPPORTED
    assert(dimension_ == 0);  // Only dimension=0 is supported for now

    // Copy each table element's raw data into the output
    std::string kernel = jtorch::jtorch_path + "kernels/join_table.cl";
    cl_context->useKernel(kernel.c_str(), "JoinTable1D");
    int out_offset = 0;
    for (uint32_t i = 0; i < in->tableSize(); i++) {
      Tensor<float>* cur_input = TO_TENSOR_PTR((*in)(i).get());
      cl_context->setArg(0, cur_input->storage());
      cl_context->setArg(1, TO_TENSOR_PTR(output.get())->storage());
      cl_context->setArg(2, out_offset);
      uint32_t dim = 1;
      uint32_t nelem = cur_input->nelems();
      cl_context->runKernel(jtorch::deviceid, dim, &nelem, false);

      out_offset += nelem;
    }
  }

}  // namespace jtorch
