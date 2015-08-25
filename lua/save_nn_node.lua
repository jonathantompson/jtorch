dofile(jtorch.jtorchRoot .. '/lua/save_sequential_node.lua')
dofile(jtorch.jtorchRoot .. '/lua/save_parallel_table_node.lua')
dofile(jtorch.jtorchRoot .. '/lua/save_tanh_node.lua')
dofile(jtorch.jtorchRoot .. '/lua/save_threshold_node.lua')
dofile(jtorch.jtorchRoot .. '/lua/save_linear_node.lua')
dofile(jtorch.jtorchRoot .. '/lua/save_reshape_node.lua')
dofile(jtorch.jtorchRoot .. '/lua/save_spatial_convolution_node.lua')
dofile(jtorch.jtorchRoot .. '/lua/save_spatial_convolution_cuda_node.lua')
dofile(jtorch.jtorchRoot .. '/lua/save_spatial_convolution_mm_node.lua')
dofile(jtorch.jtorchRoot .. '/lua/save_spatial_convolution_map_node.lua')

function jtorch._saveFloatTensorSafe(ofile, tensor)
  if torch.type(tensor) ~= 'torch.FloatTensor' then
    tensor = tensor:float()
  end
  -- We need to make sure that the underlining storage is the same size as the
  -- tensor header before saving to file
  local sz = 1
  for i = 1, tensor:dim() do
    sz = sz * tensor:size(i)
  end
  assert(tensor:storage():size() == sz, 
    'underlining storage is not the same size as the tensor')
  ofile:writeFloat(tensor:storage())
end

dofile(jtorch.jtorchRoot .. '/lua/save_spatial_lp_pooling_node.lua')
dofile(jtorch.jtorchRoot .. '/lua/save_spatial_max_pooling_node.lua')
dofile(jtorch.jtorchRoot .. 
  '/lua/save_spatial_subtractive_normalization_node.lua')
dofile(jtorch.jtorchRoot .. '/lua/save_spatial_divisive_normalization_node.lua')
dofile(jtorch.jtorchRoot .. 
  '/lua/save_spatial_contrastive_normalization_node.lua')
dofile(jtorch.jtorchRoot .. '/lua/save_join_table_node.lua')
dofile(jtorch.jtorchRoot .. '/lua/save_transpose_node.lua')
dofile(jtorch.jtorchRoot .. '/lua/save_identity_node.lua')
dofile(jtorch.jtorchRoot .. '/lua/save_select_table_node.lua')
dofile(jtorch.jtorchRoot .. '/lua/save_spatial_up_sampling_nearest_node.lua')
dofile(jtorch.jtorchRoot .. '/lua/save_c_add_table_node.lua')
dofile(jtorch.jtorchRoot .. '/lua/save_spatial_dropout.lua')

function jtorch._saveNNNode(node, ofile)
  -- Just send the node off to the correct routine depending on it's type
  -- Note that the type enum must match 
  -- jtorch/torch_stage.h
  local class_str = torch.typename(node)
  if (class_str == "nn.Sequential") then
     ofile:writeInt(1)
     jtorch._saveSequentialNode(node, ofile)
  elseif (class_str == "nn.ParallelTable") then
     ofile:writeInt(2)
     jtorch._saveParallelTableNode(node, ofile)
  elseif (class_str == "nn.Tanh") then
     ofile:writeInt(3)
     jtorch._saveTanhNode(node, ofile)
  elseif (class_str == "nn.Threshold") then
     ofile:writeInt(4)
     jtorch._saveThresholdNode(node, ofile)
  elseif (class_str == "nn.ReLU") then
     -- Note: nn.ReLU gets saved with same index as nn.Threshold
     ofile:writeInt(4)
     jtorch._saveThresholdNode(node, ofile)
  elseif (class_str == 'cudnn.ReLU') then
     -- Note: cudnn.ReLU gets saved with same index as nn.Threshold
     ofile:writeInt(4)
     node.threshold = 0  -- These don't exist for cudnn.ReLU
     node.val = 0
     jtorch._saveThresholdNode(node, ofile)
  elseif (class_str == "nn.Linear") then
     ofile:writeInt(5)
     jtorch._saveLinearNode(node, ofile)
  elseif (class_str == "nn.Reshape") then
     ofile:writeInt(6)
     jtorch._saveReshapeNode(node, ofile)
  elseif (class_str == "nn.SpatialConvolution") then
     ofile:writeInt(7)
     jtorch._saveSpatialConvolutionNode(node, ofile)
  elseif (class_str == "nn.SpatialConvolutionCUDA") then
     -- Note: SpatialConvolutionCUDA gets saved with same index
     --       as SpatialConvolution
     ofile:writeInt(7)
     jtorch._saveSpatialConvolutionCUDANode(node, ofile)
  elseif (class_str == "nn.SpatialConvolutionMap") then
     ofile:writeInt(8)
     jtorch._saveSpatialConvolutionMapNode(node, ofile)
  elseif (class_str == "nn.SpatialLPPooling") then
     ofile:writeInt(9)
     jtorch._saveSpatialLPPoolingNode(node, ofile)
  elseif (class_str == "nn.SpatialMaxPooling") then
     ofile:writeInt(10)
     jtorch._saveSpatialMaxPoolingNode(node, ofile)
  elseif (class_str == "nn.SpatialMaxPoolingCUDA") then
     -- Note: SpatialMaxPoolingCUDA gets saved with the same index
     --       as SpatialMaxPooling
     ofile:writeInt(10)
     jtorch._saveSpatialMaxPoolingNode(node, ofile)
  elseif ((class_str == "nn.SpatialSubtractiveNormalization") or 
          (class_str == "nn.SpatialSubtractiveNormalizationBatch")) then
     ofile:writeInt(11)
     jtorch._saveSpatialSubtractiveNormalizationNode(node, ofile)
  elseif ((class_str == "nn.SpatialDivisiveNormalization") or 
          (class_str == "nn.SpatialDivisiveNormalizationBatch")) then
     ofile:writeInt(12)
     jtorch._saveSpatialDivisiveNormalizationNode(node, ofile)
  elseif ((class_str == "nn.SpatialContrastiveNormalization") or 
          (class_str == "nn.SpatialContrastiveNormalizationBatch")) then
     ofile:writeInt(13)
     jtorch._saveSpatialContrastiveNormalizationNode(node, ofile)
  elseif (class_str == "nn.JoinTable") then
     ofile:writeInt(14)
     jtorch._saveJoinTableNode(node, ofile)
  elseif (class_str == "nn.Transpose") then
     ofile:writeInt(15)
     jtorch._saveTransposeNode(node, ofile)
  elseif (class_str == "nn.Identity") then
     ofile:writeInt(16)
     jtorch._saveIdentityNode(node, ofile)    
  elseif (class_str == "nn.SelectTable") then
     ofile:writeInt(17)
     jtorch._saveSelectTableNode(node, ofile)
  elseif (class_str == "nn.SpatialUpSamplingNearest") then
     ofile:writeInt(18)
     jtorch._saveSpatialUpSamplingNearestNode(node, ofile)
  elseif (class_str == "nn.CAddTable") then
     ofile:writeInt(19)
     jtorch._saveCAddTableNode(node, ofile)
  elseif ((class_str == "nn.SpatialConvolutionMM") or 
          (class_str == "nn.SpatialConvolutionMMOut")) then
     ofile:writeInt(20)
     jtorch._saveSpatialConvolutionMMNode(node, ofile)
  elseif (class_str == 'nn.SpatialDropout') then
     ofile:writeInt(21)
     jtorch._saveSpatialDropoutNode(node, ofile)
  else
     error('Node type ' .. class_str .. ' is not recognized.')
     return
  end
end

