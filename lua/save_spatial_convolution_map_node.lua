function saveSpatialConvolutionMapNode(node, ofile)

  -- The layout is as follows:
  -- 1. filter width (int)
  -- 2. filter height (int)
  -- 3. filter input features (int)
  -- 4. filter output features (int)
  -- 5. filter fan in (int)
  -- 6. filter weights (float array)
  -- 7. filter connection table (short array)
  -- 8. filter Biases (float)

  ofile:writeInt(node.kW)
  ofile:writeInt(node.kH)
  ofile:writeInt(node.nInputPlane)
  ofile:writeInt(node.nOutputPlane)

  local fanin = node.connTableRev:size()[2]
  ofile:writeInt(fanin)
 
  -- TODO: Vectorize this
  for i=1,(node.nOutputPlane * fanin) do
    for v=1,node.kH do
      for u=1,node.kW do
        ofile:writeFloat(node.weight[{i, v, u}])
      end
    end
  end
  
  -- TODO: Vectorize this
  for i=1,node.nOutputPlane do
    for v=1,fanin do
      ofile:writeShort(node.connTableRev[{i, v, 1}] - 1)  -- input feature
      ofile:writeShort(node.connTableRev[{i, v, 2}] - 1)  -- weight matrix
    end
  end

  assert(node.bias:dim() == 1, 'bias vector is not 1D!')
  saveFloatTensorSafe(ofile, node.bias)

end
