//
//  torch_stage.h
//
//  Created by Jonathan Tompson on 4/1/13.
//
//  C++ replica of various torch stages.  This is the base class that others
//  derive from.  
//
//  NOTE: YOU MUST CALL jtorch::InitTorch() before using any of these functions
//  since a valid OpenCL context must exist.
//

#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>

namespace jtorch {

  typedef enum {
    UNDEFINED_STAGE = 0,
    SEQUENTIAL_STAGE = 1,
    PARALLEL_TABLE_STAGE = 2,
    TANH_STAGE = 3,
    THRESHOLD_STAGE = 4,
    LINEAR_STAGE = 5,
    RESHAPE_STAGE = 6,
    SPATIAL_CONVOLUTION_STAGE = 7,
    SPATIAL_CONVOLUTION_MAP_STAGE = 8,
    SPATIAL_LP_POOLING_STAGE = 9,
    SPATIAL_MAX_POOLING_STAGE = 10,
    SPATIAL_SUBTRACTIVE_NORMALIZATION_STAGE = 11,
    SPATIAL_DIVISIVE_NORMALIZATION_STAGE = 12,
    SPATIAL_CONTRASTIVE_NORMALIZATION_STAGE = 13,
    JOIN_TABLE_STAGE = 14,
    TRANSPOSE_STAGE = 15,  // Not actually implemented --> Ignored
    IDENTITY_STAGE = 16,
    SELECT_TABLE_STAGE = 17,
    SPATIAL_UP_SAMPLING_NEAREST_STAGE = 18,
    C_ADD_TABLE_STAGE = 19,
  } TorchStageType;

  class TorchData;
  
  class TorchStage {
  public:
    // Constructor / Destructor
    TorchStage();
    virtual ~TorchStage();

    virtual TorchStageType type() const { return UNDEFINED_STAGE; }
    virtual void forwardProp(TorchData& input) = 0;  // Pure virtual

    // Top level read-write
    static TorchStage* loadFromFile(const std::string& file);

    // Everyone must define an output structure
    TorchData* output;

  protected:
    static TorchStage* loadFromFile(std::ifstream& file);

    // Non-copyable, non-assignable.
    TorchStage(TorchStage&);
    TorchStage& operator=(const TorchStage&);
  };

};  // namespace jtorch
