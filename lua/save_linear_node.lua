function saveLinearNode(node, ofile)
  local noutputs = node.weight:size()[1]
  local ninputs = node.weight:size()[2]
  ofile:writeInt(noutputs)
  ofile:writeInt(ninputs)
 
  assert(node.weight:dim() == 2, 'weight matrix is not 2D!')
  
  -- This could be faster with vector notation, but is OK for now
  for i=1,ninputs do
    for v=1,noutputs do
      -- weight is [nout, nin]  --> we want to store this column major
      -- So each column is contiguous and is size nout
      ofile:writeFloat(node.weight[{v, i}])  
    end
  end
  -- saveFloatTensorSafe(ofile, node.weight)  -- DOESN'T WORK! (even if I transpose it)
  
  assert(node.bias:dim() == 1, 'bias vector is not 1D!')
  saveFloatTensorSafe(ofile, node.bias)
end
