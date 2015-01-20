function saveSpatialConvolutionMMNode(node, ofile)
  -- The layout is as follows:
  -- 1. filter width (int)
  -- 2. filter height (int)
  -- 3. filter input features (int)
  -- 4. filter output features (int)
  -- 5. filter weights (float array)
  -- 6. filter Biases (float)

  ofile:writeInt(node.kW)
  ofile:writeInt(node.kH)
  ofile:writeInt(node.nInputPlane)
  ofile:writeInt(node.nOutputPlane)

  local fanin = node.nInputPlane

  assert(node.weight:dim() == 2, 'weight tensor is not 2D!')
  assert(node.weight:size(2) == (node.nInputPlane * node.kH * node.kW), 
    'bad weight tensor size!')
  -- Resize to 4D
  node.weight:resize(node.nOutputPlane, node.nInputPlane, node.kH, node.kW)
  saveFloatTensorSafe(ofile, node.weight)
  -- Now Resize back to 2D
  node.weight:resize(node.nOutputPlane, node.nInputPlane * node.kH * node.kW)
  assert(node.bias:dim() == 1, 'bias vector is not 1D!')
  saveFloatTensorSafe(ofile, node.bias)

end
