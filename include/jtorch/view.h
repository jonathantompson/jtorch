//
//  view.h
//
//  Created by Jonathan Tompson on 1/1/2016.
//
//  Partial support. We only support contiguous views where numel is same.
//

#pragma once

#include "jcl/math/math_types.h"
#include "jtorch/torch_stage.h"

namespace jtorch {

class View : public TorchStage {
 public:
  // Constructor / Destructor
  View(const uint32_t dim, const uint32_t* size);
  ~View() override;

  TorchStageType type() const override { return VIEW_STAGE; }
  std::string name() const override { return "View"; }
  void forwardProp(std::shared_ptr<TorchData> input) override;

  static std::unique_ptr<TorchStage> loadFromFile(std::ifstream& file);

 protected:
  uint32_t odim_;
  std::unique_ptr<uint32_t[]> osize_;
  void init(std::shared_ptr<TorchData> input);

  uint32_t outNElem() const;

  // Non-copyable, non-assignable.
  View(const View&) = delete;
  View& operator=(const View&) = delete;
};

};  // namespace jtorch
